using ReplayDecoder;
using System.Text.Json;

// Head position ground truth for the analyzer's wall handling (beatleader-analyzer's WallClassifier), from real replays.
//
//   BLReplaysPrepare heads <file with leaderboard ids, one per line> [--top 10]
//
// For each leaderboard, downloads the top N nomods replays and writes heads\<leaderboardId>.json next to the replays
// folders: the head position 20 times a second (time, x left/right, y height, in meters), the player's height setting and
// the walls the player touched. portaBLe compares it with the walls of the map (dodges: head moves sideways, crouches:
// head goes down).
public partial class Program
{
    public class HeadReplay
    {
        public string Player { get; set; } = "";
        public float Accuracy { get; set; }
        public float Height { get; set; }
        public List<float> Time { get; set; } = new();
        public List<float> X { get; set; } = new();
        public List<float> Y { get; set; } = new();
        // [time, energy] of every wall touched
        public List<float[]> WallHits { get; set; } = new();
    }

    public class HeadFile
    {
        public string LeaderboardId { get; set; } = "";
        public List<HeadReplay> Replays { get; set; } = new();
    }

    const float HeadSampleSeconds = 0.05f;

    public static async Task ExtractHeads(string listFile, int top)
    {
        var ids = File.ReadAllLines(listFile)
            .Select(l => (l.Contains('#') ? l[..l.IndexOf('#')] : l).Trim())
            .Where(l => l.Length > 0)
            .Distinct()
            .ToList();
        string root = ReplaysPath("heads");
        Directory.CreateDirectory(root);
        Console.WriteLine($"Heads: {ids.Count} leaderboards, top {top} replays each");

        int done = 0;
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

                var file = new HeadFile { LeaderboardId = id };
                var results = await Task.WhenAll(picked.Select(async score =>
                {
                    await replaySlots.WaitAsync();
                    try
                    {
                        var bytes = await DownloadBytes(score.Replay);
                        if (bytes == null) return null;
                        var (replay, _) = ReplayDecoder.ReplayDecoder.Decode(bytes);
                        if (replay == null || replay.frames.Count == 0) return null;
                        var head = new HeadReplay
                        {
                            Player = score.PlayerId ?? "",
                            Accuracy = (float)score.Accuracy,
                            Height = replay.info.height,
                            WallHits = replay.walls.Select(w => new[] { MathF.Round(w.time, 3), MathF.Round(w.energy, 3) }).ToList(),
                        };
                        float next = float.MinValue;
                        foreach (var frame in replay.frames)
                        {
                            if (frame.head == null || frame.time < next) continue;
                            head.Time.Add(MathF.Round(frame.time, 3));
                            head.X.Add(MathF.Round(frame.head.position.x, 3));
                            head.Y.Add(MathF.Round(frame.head.position.y, 3));
                            next = frame.time + HeadSampleSeconds;
                        }
                        return head;
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
                file.Replays = results.Where(r => r != null && r.Time.Count > 0).Select(r => r!).ToList();

                await File.WriteAllTextAsync(path, JsonSerializer.Serialize(file));
                int n = Interlocked.Increment(ref done);
                Console.WriteLine($"Heads {n}/{ids.Count}: {id}, {file.Replays.Count} replays");
            }
            catch (Exception e)
            {
                Console.WriteLine($"Heads: leaderboard {id} failed: {e.Message}");
            }
        });
    }
}
