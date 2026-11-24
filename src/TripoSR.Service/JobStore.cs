using System.Text.Json.Serialization;

namespace TripoSR.Service;

/// <summary>One model being made: what was asked, where it is, and what came out.</summary>
internal sealed class Job(string id, string engine, string detail, byte[] picture)
{
    public string Id { get; } = id;
    public string Engine { get; } = engine;
    public string Detail { get; } = detail;
    public byte[] Picture { get; set; } = picture;
    public string State { get; set; } = "queued";
    public string Error { get; set; } = "";
    public DateTime Started { get; set; }
    public DateTime Finished { get; set; }
    public string EngineVersion { get; set; } = "";
    public string DetailUsed { get; set; } = "";
    public int Size { get; set; }
    public int Triangles { get; set; }
    public byte[]? Glb { get; set; }

    public JobView View(DateTime now)
    {
        var end = Finished == default ? now : Finished;
        return new JobView(Id, State, new ProgressView(State == "done" ? 1 : 0, 1), Error, Started == default ? 0 : Math.Round((end - Started).TotalSeconds, 2),
            Engine, EngineVersion, Detail, DetailUsed, Size, Triangles);
    }
}

internal sealed record ProgressView([property: JsonPropertyName("done")] int Done, [property: JsonPropertyName("total")] int Total);

/// <summary>A job as <c>GET /jobs/{id}</c> shows it (the fields of the Python service's job, so the orchestrator reads it unchanged).</summary>
internal sealed record JobView(
    [property: JsonPropertyName("id")] string Id, [property: JsonPropertyName("state")] string State, [property: JsonPropertyName("progress")] ProgressView Progress,
    [property: JsonPropertyName("error")] string Error, [property: JsonPropertyName("seconds")] double Seconds, [property: JsonPropertyName("engine")] string Engine,
    [property: JsonPropertyName("engine_version")] string EngineVersion, [property: JsonPropertyName("detail")] string Detail,
    [property: JsonPropertyName("detail_used")] string DetailUsed, [property: JsonPropertyName("resolution")] int Resolution, [property: JsonPropertyName("triangles")] int Triangles);

/// <summary>The jobs of the service: the picture is checked when it comes, the model is made one at a time in arrival order, and a job is kept until it is forgotten.</summary>
internal sealed class JobStore(IModelEngine engine, FifoGate gate, Func<IReadOnlyList<string>>? missingFiles = null)
{
    private readonly Dictionary<string, Job> _jobs = new();
    private readonly object _lock = new();

    /// <summary>Checks a request and queues it. Throws <see cref="RefusedException"/> with the sentence for the caller (a 400) when it cannot be made.</summary>
    public Job Submit(string engineId, string detail, byte[] picture)
    {
        if (engineId != EngineInfo.Id) throw new RefusedException($"this service makes {EngineInfo.Id} models, not '{engineId}'");
        if (!EngineInfo.DetailSizes.ContainsKey(detail)) throw new RefusedException($"detail must be {string.Join(", ", EngineInfo.DetailSizes.Keys)}, got '{detail}'");
        if (picture.Length == 0) throw new RefusedException("the body must be the picture (a png, jpg or webp)");
        if (!LooksLikePicture(picture)) throw new RefusedException("the picture must be a png, jpg or webp");
        if (missingFiles?.Invoke() is { Count: > 0 } missing) throw new RefusedException($"the weights of {EngineInfo.Id} are not complete: {string.Join(", ", missing)}");
        var job = new Job(Guid.NewGuid().ToString("N")[..12], engineId, detail, picture);
        lock (_lock) _jobs[job.Id] = job;
        _ = gate.RunAsync(() => { Run(job); return 0; });
        return job;
    }

    public Job? Get(string id) { lock (_lock) return _jobs.GetValueOrDefault(id); }

    public bool Forget(string id) { lock (_lock) return _jobs.Remove(id); }

    private void Run(Job job)
    {
        job.State = "running";
        job.Started = DateTime.UtcNow;
        try
        {
            var made = engine.Generate(job.Picture, job.Detail, CancellationToken.None);
            job.Glb = made.Glb;
            (job.EngineVersion, job.DetailUsed, job.Size, job.Triangles) = (engine.Version, made.DetailUsed, made.Size, made.Triangles);
            job.State = "done";
        }
        // the error is set before the state: a client that sees "failed" must find the sentence there
        catch (RefusedException e) { job.Error = e.Message; job.State = "failed"; }
        catch (Exception e) { job.Error = $"{e.GetType().Name}: {e.Message}"; job.State = "failed"; } // the caller needs a sentence, not a trace
        finally { job.Picture = []; job.Finished = DateTime.UtcNow; }
    }

    private static bool LooksLikePicture(byte[] b) =>
        b.Length >= 12 && ((b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G')
                            || (b[0] == 0xFF && b[1] == 0xD8)
                            || (b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F' && b[8] == 'W' && b[9] == 'E' && b[10] == 'B' && b[11] == 'P'));
}
