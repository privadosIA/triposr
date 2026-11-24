using MusicGen.Core.Testing;
using TorchSharp;
using TripoSR.Field;
using TripoSR.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>The field (5.1): the MLP and the triplane query on the fixed set of points of the golden run.</summary>
public sealed class FieldTests(ITestOutputHelper output)
{
    private static readonly Lazy<NeRFMLP> Decoder = new(() =>
    {
        var m = new NeRFMLP();
        TorchCheckpointLoader.Load(m, Checkpoint.Model, CPU, "decoder.", "The checkpoint does not match the decoder:", "decoder keys", "decoder.");
        m.eval();
        return m;
    });

    public static IEnumerable<object[]> Pictures() => new[] { "chair", "hamburger", "horse", "teapot", "poly_fox" }.Select(n => new object[] { n });

    private void Within(string what, Tensor got, Tensor expected)
    {
        Assert.Equal(expected.shape, got.shape);
        var r = Parity.Compare(got, expected);
        var limit = Stage.MaxAbsLimit(expected);
        output.WriteLine($"{what}: {r} (limit {limit:E1})");
        Assert.True(r.MaxAbs <= limit && r.CosSim >= Stage.MinCosSim, $"{what}: {r} (limit {limit:E1})");
    }

    [Fact]
    public void TheDecoderCarriesTheKeysOfTheCheckpoint()
    {
        var keys = System.Text.Json.JsonSerializer.Deserialize<List<System.Text.Json.JsonElement>>(Fixtures.ReadAllText("ckpt_keys.json"))!
            .Select(k => k.GetProperty("name").GetString()!).Where(k => k.StartsWith("decoder.")).Select(k => k["decoder.".Length..]).Order(StringComparer.Ordinal).ToList();
        using var m = new NeRFMLP();
        Assert.Equal(keys, m.state_dict().Keys.Order(StringComparer.Ordinal).ToList());
        Assert.Equal(20, keys.Count);
    }

    [Fact]
    public void APointOutsideThePlanesSeesNothing()
    {
        // grid_sample pads with zeros: far outside the cube every feature is 0, so the field is the same at any two such points
        using var m = new NeRFMLP();
        using var codes = randn(3, 40, 64, 64);
        using var far = tensor(new float[] { 5, 5, 5, -7, 9, 4 }, [2, 3]);
        using var _ = no_grad();
        using var s = TriplaneField.Query(m, far, codes);
        Assert.Equal(new long[] { 2, 1 }, s.Density.shape);
        Assert.Equal(s.Color[0].data<float>().ToArray(), s.Color[1].data<float>().ToArray());
    }

    [RequiresCheckpointTheory, MemberData(nameof(Pictures))]
    public void TheFieldMatchesTheReference(string name)
    {
        var stages = Fixtures.Load(name + ".stages.safetensors");
        using var codes = stages["scene_codes"];             // (1, 3, 40, 64, 64)
        using var points = Fixtures.Load("shared.safetensors")["points"];
        using var _ = no_grad();
        using var sceneCode = codes[0];
        using var got = TriplaneField.Query(Decoder.Value, points, sceneCode);
        using var rawE = stages["field_density"];
        using var densityE = stages["field_density_act"];
        using var colorE = stages["field_color"];
        Within($"{name}: raw density", got.RawDensity, rawE);
        Within($"{name}: density", got.Density, densityE);
        Within($"{name}: colour", got.Color, colorE);
    }

    [RequiresCheckpoint]
    public void TheChunkOfPointsDoesNotChangeTheField()
    {
        var stages = Fixtures.Load("chair.stages.safetensors");
        using var codes = stages["scene_codes"];
        using var points = Fixtures.Load("shared.safetensors")["points"];
        using var _ = no_grad();
        using var sceneCode = codes[0];
        using var a = TriplaneField.Query(Decoder.Value, points, sceneCode, chunk: 8192);
        using var b = TriplaneField.Query(Decoder.Value, points, sceneCode, chunk: 1000);
        Assert.True(a.Density.allclose(b.Density, rtol: 1e-5, atol: 1e-6));
    }
}
