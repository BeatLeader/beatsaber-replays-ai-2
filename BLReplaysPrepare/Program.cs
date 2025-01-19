using System.Text.Json;
using BeatLeader_Server.Models;
using static BeatLeader_Server.Utils.ResponseUtils;
using ReplayDecoder;
using System.Net.Http.Headers;
using NumSharp;
using BeatLeader_Server.Utils;

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
    private static readonly HttpClient httpClient = new HttpClient();
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

        NDArray? output = np.zeros(new Shape(notes.Count, 3));
        int index = 0;
        foreach (var note in notes)
        {
            output[index][0] = note.noteID;
            output[index][1] = note.eventType == NoteEventType.good ? 1 - clamp(note.noteCutInfo.cutDistanceToCenter / 0.3, 0, 1) : 0.0;
            output[index][2] = note.spawnTime;
            index++;
        }

        np.save(filename, output);
    }

    public static async Task DownloadReplay(string lbFolder, float njs, ScoreResponse score) {
        if (score.Offsets == null || score.Replay?.Length < 1) return;

        try {

            var request = new HttpRequestMessage(HttpMethod.Get, score.Replay);
            request.Headers.Range = new RangeHeaderValue(score.Offsets.Notes, score.Offsets.Walls);

            HttpResponseMessage response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
            AsyncReplayDecoder decoder = new AsyncReplayDecoder();

            using (var stream1 = new MemoryStream()) {
                await (await response.Content.ReadAsStreamAsync()).CopyToAsync(stream1);
                stream1.Position = 0;
                var notes = await decoder.DecodeNotes(stream1);
                if (notes.Count == 0) {
                    request = new HttpRequestMessage(HttpMethod.Get, score.Replay);
                    response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
                    using (var stream = new MemoryStream()) {
                        await (await response.Content.ReadAsStreamAsync()).CopyToAsync(stream);
                        stream.Position = 0;
                        (var replay, var offsets) = ReplayDecoder.ReplayDecoder.Decode(stream.ToArray());

                        notes = replay?.notes ?? notes;
                    }
                }

                SaveNotes(notes, lbFolder + $"\\{score.PlayerId}-{score.Accuracy}-{score.BaseScore}-{njs}.npy");    
            }

            
        } catch (Exception e) { 
            Console.WriteLine($"Message :{e.Message}");
        }
    }

    public static async Task DownloadLeaderboardScores(string id) {
        HttpResponseMessage response = await httpClient.GetAsync($"{apiUrl}/leaderboard/scores/{id}?leaderboardContext=nomods&page=1&sortBy=rank&order=desc");
        response.EnsureSuccessStatusCode();
        string responseBody = await response.Content.ReadAsStringAsync();

        var scores = JsonSerializer.Deserialize<LeaderboardResponse>(responseBody, jsonOptions);
        
        string lbFolder = $"..\\..\\..\\..\\replays\\{id}";
        Directory.CreateDirectory(lbFolder);
        await Task.WhenAll(scores.Scores.OrderByDescending(s => s.BaseScore).Select(s => DownloadReplay(lbFolder, scores.Difficulty.Njs, s)).ToArray());
    }

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
                        HttpResponseMessage response = await httpClient.GetAsync(apiUrl + $"/leaderboards?leaderboardContext=nomods&type=all&search={song.Hash}&mode={diff.Characteristic}&difficulty={diff.Name}&page=1&count=1&sortBy=playcount&order=desc&allTypes=0&allRequirements=0");
                        response.EnsureSuccessStatusCode();
                        string responseBody = await response.Content.ReadAsStringAsync();

                        var leaderboardsPage = JsonSerializer.Deserialize<ResponseWithMetadata<LeaderboardInfoResponse>>(responseBody, jsonOptions);

                        foreach (var leaderboard in leaderboardsPage.Data)
                        {
                            if (leaderboard.Plays > 200) {
                                await DownloadLeaderboardScores(leaderboard.Id);
                                Console.WriteLine($"Leaderboard #{leaderboardCount + 1} of {totalLeaderboardCount}");
                            } else {
                                Console.WriteLine($"Skipped leaderboard #{leaderboardCount + 1} of {totalLeaderboardCount}");
                            }
                            leaderboardCount++;
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

    public static async Task Main(string[] args)
    {
        int leaderboardCount = 0;
        int pageSize = 100;
        int page = 1;
        int pageLimit = 22;

        do
        {
            try
            {
                HttpResponseMessage response = await httpClient.GetAsync(apiUrl + $"/leaderboards?leaderboardContext=nomods&page={page}&count={pageSize}&type=ranked&sortBy=playcount&order=desc&count=12&allTypes=0&allRequirements=0");
                response.EnsureSuccessStatusCode();
                string responseBody = await response.Content.ReadAsStringAsync();

                var leaderboardsPage = JsonSerializer.Deserialize<ResponseWithMetadata<LeaderboardInfo>>(responseBody, jsonOptions);

                foreach (var leaderboard in leaderboardsPage.Data)
                {
                    await DownloadLeaderboardScores(leaderboard.Id);
                    Console.WriteLine($"Leaderboard #{leaderboardCount + 1} of {leaderboardsPage.Metadata.Total}");
                    leaderboardCount++;
                }
                if (leaderboardsPage.Data.Count() < pageSize || pageLimit == page)
                {
                    break;
                }
                page++;

            } catch (HttpRequestException e)
            {
                Console.WriteLine($"\nException Caught!");
                Console.WriteLine($"Message :{e.Message}");
            }
        } while (true);

        foreach (var item in Directory.EnumerateFiles("..\\..\\..\\..\\playlists"))
        {
            await DownloadPlaylist(item);
            await Task.Delay(TimeSpan.FromSeconds(10));
        }
    }
}