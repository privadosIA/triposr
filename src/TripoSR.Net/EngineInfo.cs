namespace TripoSR;

/// <summary>What this port is, what it needs and what it answers to, in one place (the service and the tests read it).</summary>
public static class EngineInfo
{
    /// <summary>The engine id the orchestrator uses (<c>?engine=</c> of <c>POST /generate</c>).</summary>
    public const string Id = "triposr";

    /// <summary>The name shown to people.</summary>
    public const string Name = "TripoSR";

    /// <summary>The version of this port: part of what decides a model (the orchestrator caches by it), so it changes when a model would.</summary>
    public const string PortVersion = "0.1.0";

    /// <summary>What a model made by this port says made it (<c>engine_version</c> of a job and of <c>GET /engines</c>).</summary>
    public static string Version => $"{Id}-net@{PortVersion}";

    /// <summary>The files of the weights folder this engine reads, as the Python container lays them out under <c>/ckpt</c>.</summary>
    public static readonly IReadOnlyList<string> RequiredFiles = new[]
    {
        "triposr/config.yaml",
        "triposr/model.ckpt",
        "triposr/dino-vitb16-config.json",
        "rembg/u2net.onnx"
    };

    /// <summary>The details a caller may ask for and what each one is: the resolution of the marching-cubes grid (it decides the triangles).</summary>
    public static readonly IReadOnlyDictionary<string, int> DetailSizes = new Dictionary<string, int> { ["low"] = 128, ["normal"] = 192, ["high"] = 256 };

    /// <summary>The largest picture the service takes, in bytes (the Python service's limit).</summary>
    public const int MaxPictureBytes = 20 << 20;

    /// <summary>No more triangles than a scene of the orchestrator takes: a model over it is made again at the next lower detail.</summary>
    public const int MaxTriangles = 500_000;
}
