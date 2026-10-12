using BeatLeader_Server.Models;
using BeatLeader_Server.Utils;
using NumSharp;
using ReplayDecoder;
using System.Net.Http.Headers;
using System.Reflection.Emit;
using System.Text.Json;
using System.Web;
using static BeatLeader_Server.Utils.ResponseUtils;

public class Difficulty {
    public float Njs { get; set; }
}

public class LeaderboardInfo {
    public string Id { get; set; }
    public SongInfo Song { get; set; }
    public Difficulty? Difficulty { get; set; }
}

public class SongInfo {
    public string Hash { get; set; }
}

public class PlaylistDifficuly {
    public string Characteristic { get; set; }
    public string Name { get; set; }
}

public class PlaylistSong {
    public string Hash { get; set; }
    public List<PlaylistDifficuly> Difficulties { get; set; }
}

public class FilePlaylist {
    public List<PlaylistSong> Songs { get; set; }
}

public class Program
{
    // One shared client with enough connections for parallel replay downloads.
    private static readonly HttpClient httpClient = new HttpClient(new SocketsHttpHandler {
        MaxConnectionsPerServer = 64,
        AutomaticDecompression = System.Net.DecompressionMethods.All,
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
    }) { Timeout = TimeSpan.FromSeconds(60) };
    private static JsonSerializerOptions jsonOptions = new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
    public static string apiUrl = "https://api.beatleader.xyz";

    public static double clamp(double num, double min, double max) {
        return Math.Min(Math.Max(num, min), max);
    }
    public static void SaveNotes(List<NoteEvent> notes, string filename) {
        if (notes.Count == 0) return;

        // Columns: noteID, accuracy (0..1, 0 when not a good cut), spawnTime, eventType (0 good, 1 bad, 2 miss, 3 bomb).
        // The eventType column lets the trainer leave misses out of the per-note average instead of counting them as 0 accuracy.
        NDArray? output = np.zeros(new Shape(notes.Count, 4));
        int index = 0;
        foreach (var note in notes)
        {
            output[index][0] = note.noteID;
            output[index][1] = note.eventType == NoteEventType.good ? 1 - clamp(note.noteCutInfo.cutDistanceToCenter / 0.3, 0, 1) : 0.0;
            output[index][2] = note.spawnTime;
            output[index][3] = (int)note.eventType;
            index++;
        }

        lock (saveLock) {
            np.save(filename, output);
        }
    }

    // Downloads a URL (optionally a byte range) with a few retries for transient CDN errors.
    static async Task<byte[]?> DownloadBytes(string url, long? start = null, long? end = null) {
        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (start.HasValue) request.Headers.Range = new RangeHeaderValue(start, end);
                using var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
                if ((int)response.StatusCode == 404) return null;
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadAsByteArrayAsync();
            } catch (Exception e) when (e is HttpRequestException || e is TaskCanceledException || e is IOException) {
                if (attempt == 2) throw;
                await Task.Delay(TimeSpan.FromSeconds(1 + attempt * 2));
            }
        }
        return null;
    }

    public static async Task DownloadReplay(string lbFolder, float njs, ScoreResponse score) {
        try {
            if (score.Offsets == null || score.Replay?.Length < 1)
                return;

            if (!Uri.IsWellFormedUriString(score.Replay, UriKind.Absolute))
            {
                Console.WriteLine($"Invalid replay URL for {score.PlayerId}: {score.Replay}");
                return;
            }

            long? start = score.Offsets.Notes;
            long? end = score.Offsets.Walls;

            if (!start.HasValue || !end.HasValue || end.Value < start.Value)
            {
                Console.WriteLine($"Invalid offsets for {score.PlayerId}: Notes={start} Walls={end}");
                return;
            }

            // Only the notes section of the replay is downloaded (byte range); the full file is a fallback.
            var notes = new List<NoteEvent>();
            var part = await DownloadBytes(score.Replay, start, end);
            if (part != null) {
                using var stream = new MemoryStream(part);
                notes = await new AsyncReplayDecoder().DecodeNotes(stream);
            }
            if (notes.Count == 0) {
                var full = await DownloadBytes(score.Replay);
                if (full != null) {
                    (var replay, var offsets) = ReplayDecoder.ReplayDecoder.Decode(full);
                    notes = replay?.notes ?? notes;
                }
            }

            // Invariant culture so decimals are always written with '.', whatever the Windows language settings.
            SaveNotes(notes, Path.Combine(lbFolder, FormattableString.Invariant($"{score.PlayerId}-{score.Accuracy}-{score.BaseScore}-{njs}.npy")));
        } catch (Exception e) {
            Console.WriteLine($"Replay {score.PlayerId} failed: {e.Message}");
        }
    }

    public static async Task DownloadLeaderboardScores(string id) {
        string search = HttpUtility.UrlEncode(id);
        HttpResponseMessage response = await httpClient.GetAsync($"{apiUrl}/leaderboard/scores/{search}?leaderboardContext=nomods&page=1&count=50&sortBy=rank&order=desc");
        response.EnsureSuccessStatusCode();
        string responseBody = await response.Content.ReadAsStringAsync();

        var scores = JsonSerializer.Deserialize<LeaderboardResponse>(responseBody, jsonOptions);

        var ordered = scores.Scores.OrderByDescending(s => s.Accuracy).ToList();

        if (ordered.Count == 0)
        {
            Console.WriteLine($"No scores for leaderboard {id}");
            return;
        }

        List<ScoreResponse> topScores = new();
        topScores = ordered.Take(50).ToList();
        Console.WriteLine($"Downloading {topScores.Count} replays for leaderboard {id}");
        string lbFolder = $"..\\..\\..\\..\\replays\\{id}";
        Directory.CreateDirectory(lbFolder);
        await Task.WhenAll(topScores.Select(s => DownloadReplay(lbFolder, scores.Difficulty.Njs, s)).ToArray());
    }

    public static List<(string Hash, string Characteristic, string Name)> Skip = new();

    public static async Task DownloadPlaylist(string path) {
        Console.WriteLine($"Downloading playlist: {path}");
        using (StreamReader r = new StreamReader(path))
        {
            string json = r.ReadToEnd();
            var playlist = JsonSerializer.Deserialize<FilePlaylist>(json, jsonOptions);
            int leaderboardCount = 0;
            int totalLeaderboardCount = playlist.Songs.Sum(s => s.Difficulties.Count);

            foreach (var song in playlist.Songs)
            {
                foreach (var diff in song.Difficulties)
                {
                    try
                    {
                        if (Skip.Exists(x => x.Hash == song.Hash && x.Name == diff.Name && x.Characteristic == diff.Characteristic))
                        {
                            Console.WriteLine($"Skipped {song.Hash} - {diff.Characteristic} - {diff.Name}");
                        }
                        else
                        {
                            await Task.Delay(TimeSpan.FromSeconds(0.2));
                            string search = HttpUtility.UrlEncode(song.Hash);
                            string mode = HttpUtility.UrlEncode(diff.Characteristic);
                            string difficulty = HttpUtility.UrlEncode(diff.Name);
                            HttpResponseMessage response = await httpClient.GetAsync(
                                $"{apiUrl}/leaderboards?leaderboardContext=nomods&type=all&search={search}&mode={mode}&difficulty={difficulty}&page=1&count=1&sortBy=playcount&order=desc&allTypes=0&allRequirements=0"
                            ); 
                            response.EnsureSuccessStatusCode();
                            string responseBody = await response.Content.ReadAsStringAsync();

                            var leaderboardsPage = JsonSerializer.Deserialize<ResponseWithMetadata<LeaderboardInfoResponse>>(responseBody, jsonOptions);

                            foreach (var leaderboard in leaderboardsPage.Data)
                            {
                                if (leaderboard.Plays > 200)
                                {
                                    Console.WriteLine($"Leaderboard {leaderboard.Id} #{leaderboardCount + 1} of {totalLeaderboardCount}");
                                    await DownloadLeaderboardScores(leaderboard.Id);
                                }
                                else
                                {
                                    Console.WriteLine($"Skipped leaderboard {leaderboard.Id} #{leaderboardCount + 1} of {totalLeaderboardCount} due to lack of scores");
                                }
                                await Task.Delay(TimeSpan.FromSeconds(0.2));
                                Skip.Add(new(song.Hash, diff.Characteristic, diff.Name));
                                leaderboardCount++;
                            }
                        }
                    } catch (HttpRequestException e)
                    {
                        Console.WriteLine($"\nException Caught!");
                        Console.WriteLine($"Message :{e.Message}");
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Throttling
    // ------------------------------------------------------------------

    // API calls (api.beatleader.xyz) are spaced out globally so parallel leaderboards don't trip the rate limit.
    // Replay files come from the CDN and are only limited by the number of parallel downloads.
    static readonly SemaphoreSlim apiGate = new(1, 1);
    static DateTime nextApiCall = DateTime.MinValue;
    // BeatLeader allows 50 API requests per 10 s (5/s); 250 ms = 4/s leaves some headroom.
    static TimeSpan apiInterval = TimeSpan.FromMilliseconds(250);
    static SemaphoreSlim replaySlots = new(16, 16);
    static readonly object saveLock = new();
    static readonly object progressLock = new();

    static async Task WaitForApiSlot() {
        await apiGate.WaitAsync();
        try {
            var now = DateTime.UtcNow;
            if (nextApiCall > now) await Task.Delay(nextApiCall - now);
            nextApiCall = DateTime.UtcNow + apiInterval;
        } finally {
            apiGate.Release();
        }
    }

    static async Task<string> GetWithRetry(string url) {
        for (int attempt = 0; attempt < 6; attempt++) {
            await WaitForApiSlot();
            HttpResponseMessage response;
            try {
                response = await httpClient.GetAsync(url);
            } catch (Exception e) when (e is HttpRequestException || e is TaskCanceledException) {
                Console.WriteLine($"Request failed ({e.Message}), retrying");
                await Task.Delay(TimeSpan.FromSeconds(2 * (attempt + 1)));
                continue;
            }
            if ((int)response.StatusCode == 429) {
                var wait = response.Headers.RetryAfter?.Delta ?? TimeSpan.FromSeconds(5 * (attempt + 1));
                Console.WriteLine($"Rate limited, waiting {wait.TotalSeconds}s");
                // push every other API call back too, so the parallel workers don't keep hammering
                await apiGate.WaitAsync();
                try { nextApiCall = DateTime.UtcNow + wait; } finally { apiGate.Release(); }
                continue;
            }
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
        throw new HttpRequestException($"Too many retries for {url}");
    }

    static readonly HashSet<string> allowedModifiers = new() { "PM", "IF", "BE" };

    static bool HasOnlyAllowedModifiers(string? modifiers) {
        if (string.IsNullOrWhiteSpace(modifiers)) return false;
        return modifiers.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .All(m => allowedModifiers.Contains(m));
    }

    // Fetches score pages until either `maxScores` scores are collected (wide mode, maxScores > 0)
    // or the page's last score is outside the top accuracy group (tolerance mode).
    static async Task<(List<ScoreResponse> Scores, float Njs)> FetchScores(string id, string context, double tolerance, int maxScores = 0) {
        int pageSize = 100;
        var all = new List<ScoreResponse>();
        float njs = 0;
        for (int page = 1; ; page++) {
            string body = await GetWithRetry($"{apiUrl}/leaderboard/scores/{HttpUtility.UrlEncode(id)}?leaderboardContext={context}&page={page}&count={pageSize}&sortBy=rank&order=desc");
            var lb = JsonSerializer.Deserialize<LeaderboardResponse>(body, jsonOptions);
            if (njs <= 0) njs = lb?.Difficulty?.Njs ?? 0;
            var pageScores = lb?.Scores?.ToList() ?? new();
            all.AddRange(pageScores);

            if (pageScores.Count < pageSize) break;
            if (maxScores > 0) {
                if (all.Count >= maxScores) break;
            } else if (pageScores.Last().Accuracy < all.Max(s => s.Accuracy) - tolerance) {
                break;
            }
        }
        return (all, njs);
    }

    public class RunOptions {
        public string OutputFolder = "replays";
        public int MinScores = 10;
        public int MinPlays = 200;
        public int MinNotes = 20;           // leaderboards with fewer notes (BeatLeader's count) are not valid
        public double Tolerance = 0.005;
        public double MinTopAccuracy = 0.93;
        public int Top = 0;                 // > 0: wide mode, keep the top N scores whatever their spread
        public bool RecheckSkipped = false; // check again the leaderboards that were checked but not saved
        public int ParallelLeaderboards = 4;
        public int ParallelReplays = 16;
        public int ApiDelayMs = 250;
        // Leaderboards using any of these map requirements are left out (BeatLeader flags: Noodle Extensions 4,
        // Mapping Extensions 8, VNJS 128, Vivify 256, ...). 12 = Noodle + Mapping Extensions. 0 = no filter.
        public int ExcludeRequirements = 396;
    }

    // Scans the nomods leaderboards (most played first), keeps the valid ones and downloads their replays.
    // Several leaderboards are processed at once; completed ids are stored in a progress file so the run can be resumed.
    public static async Task DownloadAllValidLeaderboards(string replaysRoot, RunOptions opt) {
        Directory.CreateDirectory(replaysRoot);
        string progressFile = Path.Combine(replaysRoot, "completed.txt");
        var done = File.Exists(progressFile) ? new HashSet<string>(File.ReadAllLines(progressFile)) : new HashSet<string>();
        if (opt.RecheckSkipped)
        {
            // Only leaderboards that already have a folder count as done; everything that was checked but not saved
            // (e.g. not enough scores within the spread) is checked again with the current options.
            done = new HashSet<string>(Directory.EnumerateDirectories(replaysRoot).Select(d => Path.GetFileName(d)));
            Console.WriteLine($"Re-checking skipped leaderboards; {done.Count} saved leaderboards are kept as they are.");
        }

        var leaderboardSlots = new SemaphoreSlim(opt.ParallelLeaderboards, opt.ParallelLeaderboards);
        var running = new List<Task>();
        int processed = 0, found = 0;
        var started = DateTime.UtcNow;

        int pageSize = 5000;
        for (int page = 1; ; page++) {
            // allRequirements=2 means "not": leaderboards with any of the mapRequirements flags are excluded.
            string requirementsFilter = opt.ExcludeRequirements > 0 ? $"&mapRequirements={opt.ExcludeRequirements}&allRequirements=2" : "";
            string body = await GetWithRetry($"{apiUrl}/leaderboards?leaderboardContext=nomods&page={page}&count={pageSize}&type=all&sortBy=playcount&order=desc&allTypes=0{requirementsFilter}");
            var leaderboards = JsonSerializer.Deserialize<ResponseWithMetadata<LeaderboardInfoResponse>>(body, jsonOptions);
            var data = leaderboards?.Data?.ToList() ?? new();
            Console.WriteLine($"Page {page}: {data.Count} leaderboards");

            // The list is sorted by play count: once a whole page is under the minimum, the rest is too.
            if (data.Count > 0 && data.All(l => l.Plays < opt.MinPlays)) {
                Console.WriteLine($"Reached leaderboards with fewer than {opt.MinPlays} plays, stopping the scan.");
                break;
            }

            foreach (var lb in data) {
                if (lb.Plays < opt.MinPlays || done.Contains(lb.Id)) continue;
                if (lb.Difficulty != null && lb.Difficulty.Notes < opt.MinNotes) continue;

                await leaderboardSlots.WaitAsync();
                var id = lb.Id;
                var njs = lb.Difficulty?.Njs ?? 0;
                var label = Label(lb.Id, lb.Song?.Name, lb.Song?.Author, lb.Difficulty?.DifficultyName);
                running.Add(Task.Run(async () => {
                    try {
                        bool saved = await ProcessLeaderboard(id, njs, label, replaysRoot, opt);
                        lock (progressLock) {
                            File.AppendAllLines(progressFile, new[] { id });
                            processed++;
                            if (saved) found++;
                            if (processed % 25 == 0) {
                                var rate = processed / Math.Max(1, (DateTime.UtcNow - started).TotalMinutes);
                                Console.WriteLine($"--- {processed} leaderboards checked, {found} saved, {rate:F1}/min ---");
                            }
                        }
                    } catch (Exception e) {
                        Console.WriteLine($"Leaderboard {label} failed: {e.Message}");
                    } finally {
                        leaderboardSlots.Release();
                    }
                }));
                running.RemoveAll(t => t.IsCompleted);
            }

            if (data.Count < pageSize) break;
        }
        await Task.WhenAll(running);
        Console.WriteLine($"Finished: {processed} leaderboards checked, {found} saved.");
    }

    static async Task<bool> ProcessLeaderboard(string id, float njs, string label, string replaysRoot, RunOptions opt, int maxDownloads = int.MaxValue) {
        // Both score lists are needed, fetch them together.
        var nomodsTask = FetchScores(id, "nomods", opt.Tolerance, opt.Top);
        var generalTask = FetchScores(id, "general", opt.Tolerance, opt.Top);
        var (nomods, scoresNjs) = await nomodsTask;
        var (general, _) = await generalTask;

        // The leaderboard list sometimes has no NJS; fall back to the one from the scores endpoint.
        if (njs <= 0) njs = scoresNjs;
        if (njs <= 0)
        {
            Console.WriteLine($"Leaderboard {label} ignored: NJS unknown");
            return false;
        }

        var modded = general.Where(s => HasOnlyAllowedModifiers(s.Modifiers)).ToList();

        var candidates = nomods.Concat(modded)
            .GroupBy(s => s.PlayerId)
            .Select(g => g.OrderByDescending(s => s.Accuracy).First())
            .OrderByDescending(s => s.Accuracy)
            .ToList();

        List<ScoreResponse> valid;
        if (opt.Top > 0) {
            valid = candidates.Count >= opt.MinScores ? candidates.Take(opt.Top).ToList() : new List<ScoreResponse>();
        } else {
            valid = candidates.Count >= opt.MinScores && candidates[0].Accuracy >= opt.MinTopAccuracy
                ? candidates.TakeWhile(s => candidates[0].Accuracy - s.Accuracy <= opt.Tolerance).ToList()
                : new List<ScoreResponse>();
        }

        if (valid.Count < opt.MinScores)
        {
            Console.WriteLine($"Leaderboard {label} ignored: {valid.Count} valid scores");
            return false;
        }

        int moddedUsed = valid.Count(s => !string.IsNullOrWhiteSpace(s.Modifiers));
        Console.WriteLine($"Leaderboard {label}: downloading {Math.Min(valid.Count, maxDownloads)} replays ({moddedUsed} with modifiers)");

        string folder = Path.Combine(replaysRoot, id);
        if (Directory.Exists(folder)) Directory.Delete(folder, true);
        Directory.CreateDirectory(folder);

        // All replays of the leaderboard in parallel, capped globally by replaySlots.
        await Task.WhenAll(valid.Take(maxDownloads).Select(async score => {
            await replaySlots.WaitAsync();
            try {
                await DownloadReplay(folder, njs, score);
            } finally {
                replaySlots.Release();
            }
        }));
        return true;
    }

    // Measures the API rate limit: sends requests at increasing rates (in parallel, like the downloader)
    // and stops at the first 429. Usage: BLReplaysPrepare ratetest [leaderboardId]
    public static async Task RateTest(string leaderboardId) {
        string url = $"{apiUrl}/leaderboard/scores/{HttpUtility.UrlEncode(leaderboardId)}?leaderboardContext=nomods&page=1&count=10&sortBy=rank&order=desc";
        Console.WriteLine($"Testing {url}");

        using (var first = await httpClient.GetAsync(url)) {
            Console.WriteLine($"Status: {(int)first.StatusCode}");
            var headers = first.Headers.Concat(first.Content.Headers)
                .Where(h => h.Key.Contains("rate", StringComparison.OrdinalIgnoreCase)
                         || h.Key.Contains("limit", StringComparison.OrdinalIgnoreCase)
                         || h.Key.Contains("retry", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (headers.Count == 0) Console.WriteLine("No rate limit headers in the response.");
            foreach (var h in headers) Console.WriteLine($"  {h.Key}: {string.Join(", ", h.Value)}");
        }

        int seconds = 15, lastPassed = 0;
        foreach (int rate in new[] { 2, 4, 6, 8, 10, 15, 20, 30 }) {
            int ok = 0, limited = 0, failed = 0;
            string? retryAfter = null;
            double firstLimitedAt = -1;
            var tasks = new List<Task>();
            var sw = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < rate * seconds && Volatile.Read(ref limited) == 0; i++) {
                var due = TimeSpan.FromMilliseconds(i * 1000.0 / rate) - sw.Elapsed;
                if (due > TimeSpan.Zero) await Task.Delay(due);
                tasks.Add(Task.Run(async () => {
                    try {
                        using var r = await httpClient.GetAsync(url);
                        if ((int)r.StatusCode == 429) {
                            if (Interlocked.Increment(ref limited) == 1) {
                                firstLimitedAt = sw.Elapsed.TotalSeconds;
                                retryAfter = r.Headers.RetryAfter?.ToString();
                            }
                        } else if (r.IsSuccessStatusCode) {
                            Interlocked.Increment(ref ok);
                        } else {
                            Interlocked.Increment(ref failed);
                        }
                    } catch {
                        Interlocked.Increment(ref failed);
                    }
                }));
            }
            await Task.WhenAll(tasks);

            if (limited > 0) {
                Console.WriteLine($"{rate,3} req/s: LIMITED after {ok} OK requests ({firstLimitedAt:F1}s). Retry-After: {retryAfter ?? "none"}");
                // Stay at about half of the last rate that passed, since the downloader runs for hours.
                Console.WriteLine(lastPassed > 0
                    ? $"Suggested: --api-delay-ms {(int)Math.Ceiling(2000.0 / lastPassed)}  (about {lastPassed / 2.0:F1} req/s)"
                    : "Limited even at 2 req/s: keep --api-delay-ms 500 or more.");
                return;
            }
            Console.WriteLine($"{rate,3} req/s: {ok} OK, {failed} errors in {sw.Elapsed.TotalSeconds:F1}s");
            lastPassed = rate;
            await Task.Delay(TimeSpan.FromSeconds(30)); // let any rate window reset before the next step
        }
        Console.WriteLine("No limit hit up to 30 req/s. --api-delay-ms 100 (10 req/s) is still a polite choice.");
    }

    // Resolved from the executable location (bin\Debug\net8.0), so it doesn't depend on the working directory.
    static string ReplaysPath(string name) {
        string path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", name));
        Console.WriteLine($"Output folder: {path}");
        return path;
    }

    // Quick check on a single leaderboard
    public static async Task TestLeaderboard(string id, RunOptions opt, int maxDownloads = 3) {
        string body = await GetWithRetry($"{apiUrl}/leaderboard/scores/{HttpUtility.UrlEncode(id)}?leaderboardContext=nomods&page=1&count=1&sortBy=rank&order=desc");
        var info = JsonSerializer.Deserialize<LeaderboardResponse>(body, jsonOptions);
        float njs = info?.Difficulty?.Njs ?? 0;
        if (info?.Difficulty != null && info.Difficulty.Notes < opt.MinNotes)
            Console.WriteLine($"Note: {id} has {info.Difficulty.Notes} notes, under --min-notes {opt.MinNotes}; 'all' would skip it.");
        await ProcessLeaderboard(id, njs, Label(id, info?.Song?.Name, info?.Song?.Author, info?.Difficulty?.DifficultyName), ReplaysPath("replays_test"), opt, maxDownloads);
    }

    static string Label(string id, string? name, string? author, string? difficulty) =>
    $"{id} ({author} - {name} [{difficulty}])";

    static string? ArgValue(string[] args, string name) {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    // Usage:
    //   BLReplaysPrepare all                                  tight top groups -> replays\
    //   BLReplaysPrepare all --top 50 --out replays_wide      top 50 scores of every map, any spread
    //   BLReplaysPrepare test <leaderboardId>
    //   BLReplaysPrepare ratetest [leaderboardId]             measure the API rate limit
    //   BLReplaysPrepare all --min-scores 8 --recheck-skipped   check again only the leaderboards that weren't saved
    // Options: --parallel-leaderboards 4  --parallel-replays 16  --api-delay-ms 250  --min-plays 200  --min-scores 10
    //          --min-notes 20   (leaderboards with fewer notes are skipped)
    //          --exclude-requirements 396   (Noodle + Mapping Extensions + VNJS + Vivify; 0 to keep them)
    public static async Task Main(string[] args)
    {
        var opt = new RunOptions();
        if (ArgValue(args, "--top") is string top) opt.Top = int.Parse(top);
        if (ArgValue(args, "--out") is string output) opt.OutputFolder = output;
        if (ArgValue(args, "--parallel-leaderboards") is string pl) opt.ParallelLeaderboards = int.Parse(pl);
        if (ArgValue(args, "--parallel-replays") is string pr) opt.ParallelReplays = int.Parse(pr);
        if (ArgValue(args, "--api-delay-ms") is string ad) opt.ApiDelayMs = int.Parse(ad);
        if (ArgValue(args, "--min-plays") is string mp) opt.MinPlays = int.Parse(mp);
        if (ArgValue(args, "--min-notes") is string mn) opt.MinNotes = int.Parse(mn);
        if (ArgValue(args, "--exclude-requirements") is string er) opt.ExcludeRequirements = int.Parse(er);
        if (ArgValue(args, "--min-scores") is string ms) opt.MinScores = int.Parse(ms);
        opt.RecheckSkipped = args.Contains("--recheck-skipped");
        replaySlots = new SemaphoreSlim(opt.ParallelReplays, opt.ParallelReplays);
        apiInterval = TimeSpan.FromMilliseconds(opt.ApiDelayMs);

        if (args.Contains("ratetest")) {
            await RateTest(ArgValue(args, "ratetest") ?? "352aax11");
            return;
        }
        if (ArgValue(args, "test") is string testId) {
            await TestLeaderboard(testId, opt);
            return;
        }
        if (args.Contains("all")) {
            await DownloadAllValidLeaderboards(ReplaysPath(opt.OutputFolder), opt);
            return;
        }
        foreach (var item in Directory.EnumerateFiles("..\\..\\..\\..\\playlists"))
        {
            await DownloadPlaylist(item);
            await Task.Delay(TimeSpan.FromSeconds(1));
        }
    }
}
