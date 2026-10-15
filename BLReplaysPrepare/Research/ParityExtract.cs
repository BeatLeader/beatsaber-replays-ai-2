using ReplayDecoder;
using System.Text.Json;

// Parity ground truth for the analyzer (beatleader-analyzer's ParityPredictor), from real replays.
//
//   BLReplaysPrepare parity <file with leaderboard ids, one per line> [--top 10]
//
// For each leaderboard, downloads the top N nomods replays (full files: the hand rotations are needed) and writes
// parity2\<leaderboardId>.json next to the replays folders. For every cut note it stores the direction the saber was moving,
// expressed in the hand's own frame (an angle in the plane perpendicular to the saber). A forehand and a backhand swing
// move the saber in opposite directions relative to the hand, so this angle tells them apart without any parity guess.
// Which angle means "forehand" is calibrated later (portaBLe ParityCheck), from down arrows, which are forehands in
// practically every map.
public partial class Program
{
    public class ParityReplay
    {
        public string Player { get; set; } = "";
        public float Accuracy { get; set; }
        // One row per cut note: [note time (s), BeatLeader note id, saber (0 left, 1 right), angle in the hand frame (degrees),
        //                       world-space cut direction (degrees, 0 right / 90 up)]
        public List<float[]> Notes { get; set; } = new();
    }

    public class ParityFile
    {
        public string LeaderboardId { get; set; } = "";
        public List<ParityReplay> Replays { get; set; } = new();
    }

    public static async Task ExtractParity(string listFile, int top)
    {
        var ids = File.ReadAllLines(listFile)
            .Select(l => (l.Contains('#') ? l[..l.IndexOf('#')] : l).Trim())   // "id  # comment"
            .Where(l => l.Length > 0)
            .Distinct()
            .ToList();
        // parity2: rows also hold the world-space cut direction (parity\ has the older 4-value rows)
        string root = ReplaysPath("parity2");
        Directory.CreateDirectory(root);
        Console.WriteLine($"Parity: {ids.Count} leaderboards, top {top} replays each");

        int done = 0;
        // A few leaderboards at a time (their replays share replaySlots), so the whole ranked list finishes in a few hours
        await Parallel.ForEachAsync(ids, new ParallelOptions { MaxDegreeOfParallelism = 4 }, async (id, ct) =>
        {
            string path = Path.Combine(root, $"{id}.json");
            if (File.Exists(path))
            {
                Interlocked.Increment(ref done);
                return;
            }

            try
            {
                var (scores, _) = await FetchScores(id, "nomods", 0, top);
                var picked = scores
                    .Where(s => !string.IsNullOrWhiteSpace(s.Replay) && Uri.IsWellFormedUriString(s.Replay, UriKind.Absolute))
                    .Where(s => !HasSpeedModifier(s.Modifiers))
                    .GroupBy(s => s.PlayerId)
                    .Select(g => g.First())
                    .Take(top)
                    .ToList();

                var file = new ParityFile { LeaderboardId = id };
                var results = await Task.WhenAll(picked.Select(async score =>
                {
                    await replaySlots.WaitAsync();
                    try
                    {
                        var bytes = await DownloadBytes(score.Replay);
                        if (bytes == null) return null;
                        var (replay, _) = ReplayDecoder.ReplayDecoder.Decode(bytes);
                        if (replay == null || replay.info.leftHanded || replay.frames.Count == 0) return null;
                        return new ParityReplay
                        {
                            Player = score.PlayerId ?? "",
                            Accuracy = (float)score.Accuracy,
                            Notes = ParityRows(replay),
                        };
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"Replay {score.PlayerId} on {id} failed: {e.Message}");
                        return null;
                    }
                    finally
                    {
                        replaySlots.Release();
                    }
                }));
                file.Replays = results.Where(r => r != null && r.Notes.Count > 0).Select(r => r!).ToList();

                await File.WriteAllTextAsync(path, JsonSerializer.Serialize(file));
                int n = Interlocked.Increment(ref done);
                Console.WriteLine($"Parity {n}/{ids.Count}: {id}, {file.Replays.Count} replays");
            }
            catch (Exception e)
            {
                Console.WriteLine($"Parity: leaderboard {id} failed: {e.Message}");
            }
        });
    }

    static bool HasSpeedModifier(string? modifiers)
    {
        if (string.IsNullOrWhiteSpace(modifiers)) return false;
        return modifiers.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Any(m => m == "SS" || m == "FS" || m == "SF");
    }

    static List<float[]> ParityRows(Replay replay)
    {
        var rows = new List<float[]>();
        var frames = replay.frames;
        foreach (var note in replay.notes)
        {
            if (note.eventType != NoteEventType.good && note.eventType != NoteEventType.bad) continue;
            var cut = note.noteCutInfo;
            if (cut == null) continue;
            var dir = cut.saberDir;
            if (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z < 1e-6f) continue;

            var frame = NearestFrame(frames, note.eventTime);
            if (frame == null) continue;
            var hand = cut.saberType == 0 ? frame.leftHand : frame.rightHand;
            if (hand == null) continue;

            // Saber motion in the hand's frame: rotate by the inverse of the hand rotation
            var q = hand.rotation;
            var (lx, ly, _) = Rotate(-q.x, -q.y, -q.z, q.w, dir.x, dir.y, dir.z);
            if (lx * lx + ly * ly < 1e-6f) continue;
            float angle = (float)(Math.Atan2(ly, lx) * 180.0 / Math.PI);
            // Direction the saber moved at the cut, in the plane of the notes (x right, y up)
            float world = (float)(Math.Atan2(dir.y, dir.x) * 180.0 / Math.PI);

            rows.Add(new[] { note.spawnTime, note.noteID, cut.saberType, MathF.Round(angle, 1), MathF.Round(world, 1) });
        }
        return rows;
    }

    static Frame? NearestFrame(List<Frame> frames, float time)
    {
        if (frames.Count == 0) return null;
        int lo = 0, hi = frames.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (frames[mid].time < time) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.Abs(frames[lo - 1].time - time) < Math.Abs(frames[lo].time - time)) lo--;
        return frames[lo];
    }

    // Rotates vector v by the unit quaternion (x, y, z, w)
    static (float, float, float) Rotate(float qx, float qy, float qz, float qw, float vx, float vy, float vz)
    {
        // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
        float tx = 2 * (qy * vz - qz * vy);
        float ty = 2 * (qz * vx - qx * vz);
        float tz = 2 * (qx * vy - qy * vx);
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx));
    }
}
