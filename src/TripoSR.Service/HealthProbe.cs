using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Serialization;
using TorchSharp;

namespace TripoSR.Service;

internal sealed record GpuView([property: JsonPropertyName("name")] string Name, [property: JsonPropertyName("free_mb")] long? FreeMb, [property: JsonPropertyName("total_mb")] long? TotalMb);

internal sealed record EngineView([property: JsonPropertyName("built")] bool Built, [property: JsonPropertyName("missing")] IReadOnlyList<string> Missing,
    [property: JsonPropertyName("ready")] bool Ready, [property: JsonPropertyName("version")] string Version);

internal sealed record HealthView(
    [property: JsonPropertyName("ready")] bool Ready, [property: JsonPropertyName("version")] string Version, [property: JsonPropertyName("busy")] bool Busy,
    [property: JsonPropertyName("queued")] int Queued, [property: JsonPropertyName("gpu")] GpuView? Gpu, [property: JsonPropertyName("missing")] IReadOnlyList<string> Missing,
    [property: JsonPropertyName("engines")] IReadOnlyDictionary<string, EngineView> Engines);

/// <summary>
/// What <c>GET /health</c> and <c>GET /engines</c> say: whether the weights are there and the device works, and what card there is. The checks are cheap, cached for a few
/// seconds and never load the model; <c>busy</c> and <c>queued</c> are live, so health answers while a model is made.
/// </summary>
internal sealed class HealthProbe
{
    private readonly ServiceSettings _settings;
    private readonly FifoGate _gate;
    private readonly string _engineVersion;
    private readonly TimeSpan _cache;
    private readonly object _lock = new();
    private (IReadOnlyList<string> Missing, GpuView? Gpu)? _last;
    private DateTime _lastAt;

    // seams for tests
    public Func<IReadOnlyList<string>> MissingFiles { get; init; }
    public Func<string?> DeviceProblem { get; init; }
    public Func<GpuView?> Gpu { get; init; }

    public HealthProbe(ServiceSettings settings, FifoGate gate, string engineVersion, TimeSpan? cacheFor = null)
    {
        _settings = settings;
        _gate = gate;
        _engineVersion = engineVersion;
        _cache = cacheFor ?? TimeSpan.FromSeconds(5);
        MissingFiles = () => EngineInfo.RequiredFiles.Where(f => !File.Exists(Path.Combine(settings.CheckpointDirectory, f))).ToList();
        DeviceProblem = () => ProblemWith(settings.Device);
        Gpu = () => ReadGpu(settings.Device);
    }

    public HealthView Get()
    {
        var (missing, gpu) = State();
        return new HealthView(missing.Count == 0, _settings.Version, _gate.Busy, _gate.Queued, gpu, missing, Engines());
    }

    /// <summary>The engines this service has, as <c>GET /engines</c> shows them (one: this port).</summary>
    public IReadOnlyDictionary<string, EngineView> Engines()
    {
        var (missing, _) = State();
        return new Dictionary<string, EngineView> { [EngineInfo.Id] = new(true, missing, missing.Count == 0, _engineVersion) };
    }

    private (IReadOnlyList<string> Missing, GpuView? Gpu) State()
    {
        lock (_lock)
        {
            if (_last is null || DateTime.UtcNow - _lastAt >= _cache)
            {
                var missing = MissingFiles().ToList();
                if (DeviceProblem() is { } problem) missing.Add(problem);
                _last = (missing, Gpu());
                _lastAt = DateTime.UtcNow;
            }
            return _last.Value;
        }
    }

    public static string? ProblemWith(string device)
    {
        if (!device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)) return null;
        if (!torch.cuda.is_available()) return $"device {device} is not available (CPU build of the service, or no CUDA device)";
        var index = device.Contains(':') ? int.Parse(device[(device.IndexOf(':') + 1)..], CultureInfo.InvariantCulture) : 0;
        return index < torch.cuda.device_count() ? null : $"device {device} is not available (only {torch.cuda.device_count()} CUDA device(s))";
    }

    // ---- the card ---------------------------------------------------------------------------------------------------------------------

    private static GpuView? ReadGpu(string device)
    {
        if (!device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)) return null;
        var index = device.Contains(':') && int.TryParse(device[(device.IndexOf(':') + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out var i) ? i : 0;
        var fallback = new GpuView($"cuda:{index}", null, null);
        try
        {
            var smi = FindNvidiaSmi();
            if (smi is null) return fallback;
            using var p = Process.Start(new ProcessStartInfo(smi, $"--query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits -i {index}")
            { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false });
            if (p is null) return fallback;
            var text = p.StandardOutput.ReadToEndAsync();
            if (!p.WaitForExit(2000)) { try { p.Kill(); } catch (InvalidOperationException) { } return fallback; }
            return ParseSmi(text.Result) ?? fallback;
        }
        catch (Win32Exception) { return fallback; }
        catch (InvalidOperationException) { return fallback; }
    }

    /// <summary>The absolute path of <c>nvidia-smi</c> in its standard install locations, or null (never looked up in the PATH).</summary>
    internal static string? FindNvidiaSmi()
    {
        IEnumerable<string> candidates = OperatingSystem.IsWindows()
            ? [Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "nvidia-smi.exe"),
               Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")]
            : ["/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"];
        return candidates.FirstOrDefault(c => Path.IsPathRooted(c) && File.Exists(c));
    }

    /// <summary>Reads the first line of <c>nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits</c>.</summary>
    public static GpuView? ParseSmi(string output)
    {
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault();
        if (line is null) return null;
        var parts = line.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 3 || parts[0].Length == 0) return null;
        long? Num(string s) => long.TryParse(s, NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : null;
        return new GpuView(parts[0], Num(parts[1]), Num(parts[2]));
    }
}
