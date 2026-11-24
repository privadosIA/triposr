using MusicGen.Core.Testing;
using TripoSR.Weights;

namespace TripoSR.Net.Tests;

/// <summary>
/// Where the tests find the real weights: <c>TRIPOSR_CKPT</c> is the weights folder (the layout of the container's <c>/ckpt</c>), never inside the
/// repository. Tests that need it are marked <see cref="RequiresCheckpointAttribute"/> and are reported as skipped, with the reason, without it.
/// </summary>
public static class Checkpoint
{
    public static readonly CheckpointEnvironment Env = new("TRIPOSR_CKPT", null, "TRIPOSR_DEVICE");

    private static readonly Lazy<TorchCheckpoint> Model_ = new(() => TorchCheckpoint.Open(Path.Combine(Dir!, "triposr", "model.ckpt")));

    public static string? Dir => Env.Dir;

    /// <summary>The checkpoint <c>triposr/model.ckpt</c>, opened once (headers only).</summary>
    public static TorchCheckpoint Model => Model_.Value;
}

/// <summary>[Fact] that is skipped (saying why) unless TRIPOSR_CKPT points to an existing weights folder.</summary>
public sealed class RequiresCheckpointAttribute : RequiresCheckpointFactAttribute
{
    public RequiresCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>[Theory] that is skipped (saying why) unless TRIPOSR_CKPT points to an existing weights folder.</summary>
public sealed class RequiresCheckpointTheoryAttribute : Xunit.TheoryAttribute
{
    public RequiresCheckpointTheoryAttribute() => Skip = Checkpoint.Env.NoCheckpointReason;
}

/// <summary>The parity thresholds of a stage (openspec/config.yaml): fp32, <c>max_abs &lt;= 1e-4 * max(1, max|expected|)</c> and <c>cos_sim &gt;= 0.9999</c>.</summary>
public static class Stage
{
    public const double MinCosSim = 0.9999;

    public static double MaxAbsLimit(TorchSharp.torch.Tensor expected)
    {
        using var _ = TorchSharp.torch.no_grad();
        var scale = expected.detach().abs().max().item<float>();
        return 1e-4 * Math.Max(1.0, scale);
    }
}

/// <summary>Access to tests/fixtures (golden run of the Python reference), each file checked against the sha256 of MANIFEST.md before it is used.</summary>
public static class Fixtures
{
    private static readonly FixtureDirectory Directory_ = FixtureDirectory.Find();

    public static string Dir => Directory_.Dir;

    public static string PathOf(string file) => Directory_.PathOf(file);

    public static Dictionary<string, TorchSharp.torch.Tensor> Load(string file) => Directory_.Load(file);

    public static string ReadAllText(string file) => Directory_.ReadAllText(file);
}
