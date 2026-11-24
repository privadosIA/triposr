using System.Text.Json;

namespace TripoSR.Vision;

/// <summary>The DINO ViT-B/16 hyper-parameters (<c>triposr/dino-vitb16-config.json</c>, a Hugging Face <c>ViTConfig</c>); the defaults are that file's values.</summary>
public sealed record DinoConfig
{
    public int HiddenSize { get; init; } = 768;
    public int NumHiddenLayers { get; init; } = 12;
    public int NumAttentionHeads { get; init; } = 12;
    public int IntermediateSize { get; init; } = 3072;
    public int PatchSize { get; init; } = 16;
    public int NumChannels { get; init; } = 3;
    public int ImageSize { get; init; } = 224;
    public double LayerNormEps { get; init; } = 1e-12;
    public string HiddenAct { get; init; } = "gelu";

    public static DinoConfig FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var d = new DinoConfig();
        int Int(string name, int fallback) => r.TryGetProperty(name, out var v) ? v.GetInt32() : fallback;
        var c = d with
        {
            HiddenSize = Int("hidden_size", d.HiddenSize),
            NumHiddenLayers = Int("num_hidden_layers", d.NumHiddenLayers),
            NumAttentionHeads = Int("num_attention_heads", d.NumAttentionHeads),
            IntermediateSize = Int("intermediate_size", d.IntermediateSize),
            PatchSize = Int("patch_size", d.PatchSize),
            NumChannels = Int("num_channels", d.NumChannels),
            ImageSize = Int("image_size", d.ImageSize),
            LayerNormEps = r.TryGetProperty("layer_norm_eps", out var eps) ? eps.GetDouble() : d.LayerNormEps,
            HiddenAct = r.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? d.HiddenAct : d.HiddenAct,
        };
        if (c.HiddenAct != "gelu") throw new NotSupportedException($"{path}: hidden_act {c.HiddenAct} is not supported (the checkpoint uses gelu)");
        if (c.HiddenSize % c.NumAttentionHeads != 0) throw new InvalidDataException($"{path}: hidden_size {c.HiddenSize} is not a multiple of num_attention_heads {c.NumAttentionHeads}");
        return c;
    }
}
