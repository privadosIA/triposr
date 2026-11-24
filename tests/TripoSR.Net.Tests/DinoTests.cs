using MusicGen.Core.Testing;
using TorchSharp;
using TripoSR.Vision;
using TripoSR.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>Image tokens: DINO ViT-B/16 in C# against the tokens of the reference (3.1).</summary>
public sealed class DinoTests(ITestOutputHelper output)
{
    private const string Prefix = "image_tokenizer.";

    private static readonly Lazy<DinoSingleImageTokenizer> Tokenizer = new(() => Load(null));

    private static DinoSingleImageTokenizer Load(DinoConfig? config)
    {
        var cfg = config ?? DinoConfig.FromFile(Path.Combine(Checkpoint.Dir!, "triposr", "dino-vitb16-config.json"));
        var m = new DinoSingleImageTokenizer(cfg);
        TorchCheckpointLoader.Load(m, Checkpoint.Model, CPU, Prefix, "The checkpoint does not match the image tokenizer:", "image tokenizer keys", Prefix);
        m.eval();
        return m;
    }

    public static IEnumerable<object[]> Pictures() => new[] { "chair", "hamburger", "horse", "teapot", "poly_fox" }.Select(n => new object[] { n });

    [Fact]
    public void TheConfigIsTheOneOfTheFile()
    {
        var c = new DinoConfig();
        Assert.Equal((768, 12, 12, 3072, 16, 1e-12), (c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads, c.IntermediateSize, c.PatchSize, c.LayerNormEps));
    }

    [Fact]
    public void TheModulesCarryTheKeysOfTheCheckpoint()
    {
        // the reference's keys of the image tokenizer, without the prefix of the module: 200 tensors, none of them a buffer
        var keys = System.Text.Json.JsonSerializer.Deserialize<List<System.Text.Json.JsonElement>>(Fixtures.ReadAllText("ckpt_keys.json"))!
            .Select(k => k.GetProperty("name").GetString()!).Where(k => k.StartsWith(Prefix)).Select(k => k[Prefix.Length..]).Order(StringComparer.Ordinal).ToList();
        using var m = new DinoSingleImageTokenizer();
        var mine = m.state_dict().Keys.Order(StringComparer.Ordinal).ToList();
        Assert.Equal(keys, mine);
    }

    [RequiresCheckpoint]
    public void ItLoadsStrictly() => Assert.NotNull(Tokenizer.Value);

    [RequiresCheckpoint]
    public void AModelWithAnotherShapeIsRefusedNamingTheKeys()
    {
        var e = Assert.Throws<InvalidDataException>(() => Load(new DinoConfig { NumHiddenLayers = 11 }));
        Assert.Contains("model.encoder.layer.11.attention.attention.query.weight", e.Message);
    }

    [RequiresCheckpointTheory, MemberData(nameof(Pictures))]
    public void TheTokensMatchTheReference(string name)
    {
        var stages = Fixtures.Load(name + ".stages.safetensors");
        using var rgb = stages["rgb_cond"];                  // (1, 512, 512, 3)
        using var images = rgb.permute(0, 3, 1, 2).contiguous(); // (1, 3, 512, 512)
        using var expected = stages["image_tokens"];         // (1, 1025, 768)
        using var _ = no_grad();
        using var got = Tokenizer.Value.forward(images);
        Assert.Equal(expected.shape, got.shape);
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{name}: {r}");
        Assert.True(r.MaxAbs <= Stage.MaxAbsLimit(expected) && r.CosSim >= Stage.MinCosSim, r.ToString());
    }
}
