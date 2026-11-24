using MusicGen.Core.Testing;
using TorchSharp;
using TripoSR.Mesh;
using TripoSR.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>The whole network: the strict load of the checkpoint (1.2) and a picture to a model (7.1), against the container's models.</summary>
public sealed class PipelineTests(ITestOutputHelper output)
{
    private static readonly Lazy<TripoSRPipeline> Pipeline = new(() => TripoSRPipeline.Load(Checkpoint.Dir!));

    public static IEnumerable<object[]> Pictures() => new[] { "chair", "hamburger", "horse", "teapot", "poly_fox" }.Select(n => new object[] { n });

    private static string Picture(string name) => Directory.GetFiles(Path.Combine(Fixtures.Dir, "pictures"), name + ".*").Single();

    [Fact]
    public void TheModulesOfTheWholeNetworkAreTheKeysOfTheCheckpoint()
    {
        var keys = System.Text.Json.JsonSerializer.Deserialize<List<System.Text.Json.JsonElement>>(Fixtures.ReadAllText("ckpt_keys.json"))!
            .Select(k => k.GetProperty("name").GetString()!).Order(StringComparer.Ordinal).ToList();
        using var m = new TripoSRModel();
        Assert.Equal(keys, m.state_dict().Keys.Order(StringComparer.Ordinal).ToList());
        Assert.Equal(549, keys.Count);
    }

    [RequiresCheckpoint]
    public void TheCheckpointLoadsStrictlyEveryKeyUsedOnceAndNoneLeftOver()
    {
        using var m = new TripoSRModel();
        TorchCheckpointLoader.Load(m, Checkpoint.Model, CPU, "", "The checkpoint does not match the TripoSR network:", "checkpoint keys");
        var state = m.state_dict();
        Assert.Equal(Checkpoint.Model.Tensors.Count, state.Count);
        foreach (var (key, info) in Checkpoint.Model.Tensors.Take(40))
        {
            using var expected = Checkpoint.Model.Read(key);
            Assert.True(state[key].eq(expected).all().item<bool>(), key);
        }
    }

    [RequiresCheckpoint]
    public void ANetworkOfAnotherShapeIsRefusedNamingTheKeys()
    {
        using var m = new TripoSRModel(new TripoSRConfig { Layers = 15 });
        var e = Assert.Throws<InvalidDataException>(() => TorchCheckpointLoader.Load(m, Checkpoint.Model, CPU, "", "The checkpoint does not match the TripoSR network:", "checkpoint keys"));
        Assert.Contains("backbone.transformer_blocks.15.attn1.to_q.weight", e.Message);
    }

    [Fact]
    public void AConfigOfAnotherNetworkIsRefused()
    {
        var yaml = "cond_image_size: 512\nnum_layers: 24\n";
        var e = Assert.Throws<InvalidDataException>(() => new TripoSRConfig().Verify(yaml, "triposr/config.yaml"));
        Assert.Contains("num_layers is 24, this port is 16", e.Message);
        Assert.Contains("triposr/config.yaml", e.Message);
    }

    [RequiresCheckpoint]
    public void TheConfigOfTheWeightsIsTheNetworkOfThisPort() =>
        new TripoSRConfig().Verify(File.ReadAllText(Path.Combine(Checkpoint.Dir!, "triposr", "config.yaml")));

    [RequiresCheckpointTheory, MemberData(nameof(Pictures))]
    public void ThePictureBecomesTheModelOfTheContainer(string name)
    {
        var made = Pipeline.Value.Make(File.ReadAllBytes(Picture(name)), "low");
        var glb = GlbWriter.Write(made.Mesh);
        var reference = MeshCompare.ReadGlb(File.ReadAllBytes(Fixtures.PathOf(name + ".low.glb")));
        var (got, _) = MeshCompare.ReadGlb(glb);
        Assert.Equal(("low", 128), (made.DetailUsed, made.Size));

        var triangleDifference = Math.Abs(made.Mesh.TriangleCount - reference.Triangles) / (double)reference.Triangles;
        var (glo, ghi) = MeshCompare.Bounds(got);
        var (rlo, rhi) = MeshCompare.Bounds(reference.Vertices);
        double size = Math.Sqrt(Enumerable.Range(0, 3).Sum(k => (rhi[k] - rlo[k]) * (rhi[k] - rlo[k])));
        var boxDifference = Enumerable.Range(0, 3).Max(k => Math.Max(Math.Abs(glo[k] - rlo[k]), Math.Abs(ghi[k] - rhi[k]))) / size;
        var chamfer = MeshCompare.Chamfer(got, reference.Vertices);
        output.WriteLine($"{name}: triangles {made.Mesh.TriangleCount} (reference {reference.Triangles}, {triangleDifference:P3}), bounding box differs {boxDifference:E2} of the size, Chamfer {chamfer:E2} of the size");
        Assert.True(triangleDifference <= 0.01, $"triangles differ {triangleDifference:P2}");
        Assert.True(boxDifference <= 0.01, $"the bounding box differs {boxDifference:P2} of its size");
        Assert.True(chamfer < 1e-3, $"Chamfer distance {chamfer:E2} of the size");
        Assert.Equal(0.0, glo[1]);   // stands on the floor
    }

    [RequiresCheckpoint]
    public void APictureThatIsNotOneIsRefusedWithASentence()
    {
        var e = Assert.Throws<ModelRefusedException>(() => Pipeline.Value.Make([1, 2, 3, 4], "low"));
        Assert.Contains("not a picture", e.Message);
    }
}
