using System.Text.RegularExpressions;
using MusicGen.Core.Testing;
using TorchSharp;
using TripoSR.Transformer;
using TripoSR.Triplane;
using TripoSR.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>The triplane tokens, the backbone and the scene codes (4.1, 4.2, 4.3) against the golden run.</summary>
public sealed class TriplaneBackboneTests(ITestOutputHelper output)
{
    private static readonly Lazy<TriplaneTokenizer> Tokenizer = new(() => Load(new TriplaneTokenizer(), "tokenizer."));
    private static readonly Lazy<Transformer1D> Backbone = new(() => Load(new Transformer1D(), "backbone."));
    private static readonly Lazy<TriplaneUpsampleNetwork> PostProcessor = new(() => Load(new TriplaneUpsampleNetwork(), "post_processor."));

    private static T Load<T>(T module, string prefix) where T : nn.Module
    {
        TorchCheckpointLoader.Load(module, Checkpoint.Model, CPU, prefix, $"The checkpoint does not match {prefix.TrimEnd('.')}:", $"{prefix.TrimEnd('.')} keys", prefix);
        module.eval();
        return module;
    }

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
    public void TheConfigIsTheOneOfConfigYaml()
    {
        var path = Path.Combine(Fixtures.Dir, "..", "..", "src", "TripoSR.Net", "TripoSRConfig.cs");
        Assert.True(File.Exists(path));
        // config.yaml lives in the weights folder (never in the repo): its numbers are checked when it is there, and are these otherwise
        var c = new TripoSRConfig();
        Assert.Equal((512, 32, 1024, 16, 64, 16, 768, 40, 64, 9), (c.CondImageSize, c.PlaneSize, c.TokenChannels, c.Heads, c.HeadDim, c.Layers, c.CrossAttentionDim, c.SceneChannels, c.DecoderHidden, c.DecoderHiddenLayers));
        Assert.Equal(0.87, c.Radius);
    }

    [RequiresCheckpoint]
    public void TheConfigMatchesTheFileOfTheWeights()
    {
        var yaml = File.ReadAllText(Path.Combine(Checkpoint.Dir!, "triposr", "config.yaml"));
        int Num(string key) => int.Parse(Regex.Match(yaml, @"^\s*" + key + @":\s*(\d+)", RegexOptions.Multiline).Groups[1].Value);
        var c = new TripoSRConfig();
        Assert.Equal(c.CondImageSize, Num("cond_image_size"));
        Assert.Equal(c.PlaneSize, Num("plane_size"));
        Assert.Equal(c.TokenChannels, Num("num_channels"));
        Assert.Equal(c.Heads, Num("num_attention_heads"));
        Assert.Equal(c.HeadDim, Num("attention_head_dim"));
        Assert.Equal(c.Layers, Num("num_layers"));
        Assert.Equal(c.CrossAttentionDim, Num("cross_attention_dim"));
        Assert.Equal(c.SceneChannels, Num("out_channels"));
        Assert.Equal(c.DecoderHidden, Num("n_neurons"));
        Assert.Equal(c.DecoderHiddenLayers, Num("n_hidden_layers"));
        Assert.Contains("radius: 0.87", yaml);
        Assert.Contains("density_bias: -1.0", yaml);
    }

    [Fact]
    public void TheModulesCarryTheKeysOfTheCheckpoint()
    {
        var keys = System.Text.Json.JsonSerializer.Deserialize<List<System.Text.Json.JsonElement>>(Fixtures.ReadAllText("ckpt_keys.json"))!
            .Select(k => k.GetProperty("name").GetString()!).ToList();
        void Check(nn.Module m, string prefix)
        {
            var expected = keys.Where(k => k.StartsWith(prefix)).Select(k => k[prefix.Length..]).Order(StringComparer.Ordinal).ToList();
            Assert.Equal(expected, m.state_dict().Keys.Order(StringComparer.Ordinal).ToList());
        }
        using (var m = new TriplaneTokenizer()) Check(m, "tokenizer.");
        using (var m = new Transformer1D()) Check(m, "backbone.");
        using (var m = new TriplaneUpsampleNetwork()) Check(m, "post_processor.");
    }

    [Fact]
    public void TheTokensHaveTheLayoutOfTheReference()
    {
        // tokenizer then detokenize is the identity on the embedding, and the layout is B Ct (Np Hp Wp)
        using var m = new TriplaneTokenizer();
        using var emb = m.parameters().Single().detach();
        using var _ = no_grad();
        emb.copy_(arange(3 * 1024 * 32 * 32, dtype: ScalarType.Float32).reshape(3, 1024, 32, 32));
        using var tokens = m.forward(2);
        Assert.Equal(new long[] { 2, 1024, 3072 }, tokens.shape);
        Assert.Equal(emb[1, 5, 3, 7].item<float>(), tokens[1, 5, 1 * 1024 + 3 * 32 + 7].item<float>());
        using var planes = m.detokenize(tokens);
        Assert.Equal(new long[] { 2, 3, 1024, 32, 32 }, planes.shape);
        Assert.True(planes[1].allclose(emb));
    }

    [RequiresCheckpoint]
    public void TheTriplaneTokensMatchTheReference()
    {
        using var expected = Fixtures.Load("shared.safetensors")["triplane_tokens"];
        using var _ = no_grad();
        using var got = Tokenizer.Value.forward(1);
        Within("triplane tokens", got, expected);
    }

    [RequiresCheckpointTheory, MemberData(nameof(Pictures))]
    public void TheBackboneMatchesTheReference(string name)
    {
        var stages = Fixtures.Load(name + ".stages.safetensors");
        using var imageTokens = stages["image_tokens"];
        using var expected = stages["backbone"];
        using var tokens = Fixtures.Load("shared.safetensors")["triplane_tokens"];
        using var _ = no_grad();
        using var got = Backbone.Value.forward(tokens, imageTokens);
        Within($"{name}: backbone", got, expected);
    }

    [RequiresCheckpointTheory, MemberData(nameof(Pictures))]
    public void TheSceneCodesMatchTheReference(string name)
    {
        var stages = Fixtures.Load(name + ".stages.safetensors");
        using var backbone = stages["backbone"];
        using var expected = stages["scene_codes"];
        using var _ = no_grad();
        using var planes = Tokenizer.Value.detokenize(backbone);
        using var got = PostProcessor.Value.forward(planes);
        Within($"{name}: scene codes", got, expected);
    }
}
