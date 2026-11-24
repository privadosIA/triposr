using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TripoSR.Vision;

// The field names below are the keys of the checkpoint (image_tokenizer.model.*, the Hugging Face ViTModel's names), so loading needs no mapping.
#pragma warning disable IDE1006 // naming: the names are the checkpoint's

/// <summary>The patch embedding of the ViT: a convolution of the patches into <c>HiddenSize</c> channels.</summary>
public sealed class DinoPatchEmbeddings : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> projection;

    public DinoPatchEmbeddings(DinoConfig c) : base(nameof(DinoPatchEmbeddings))
    {
        projection = nn.Conv2d(c.NumChannels, c.HiddenSize, c.PatchSize, c.PatchSize);
        RegisterComponents();
    }

    /// <summary>(B, 3, H, W) to (B, H/16 * W/16, hidden).</summary>
    public override Tensor forward(Tensor pixelValues) => projection.forward(pixelValues).flatten(2).transpose(1, 2);
}

/// <summary>Class token, patch embeddings and the position encoding, interpolated (bicubic) to the picture's size, as <c>interpolate_pos_encoding=True</c> does.</summary>
public sealed class DinoEmbeddings : nn.Module<Tensor, Tensor>
{
    private readonly Parameter cls_token;
    private readonly Parameter position_embeddings;
    private readonly DinoPatchEmbeddings patch_embeddings;
    private readonly int _patch;

    public DinoEmbeddings(DinoConfig c) : base(nameof(DinoEmbeddings))
    {
        _patch = c.PatchSize;
        var positions = (c.ImageSize / c.PatchSize) * (c.ImageSize / c.PatchSize) + 1;
        cls_token = nn.Parameter(zeros(1, 1, c.HiddenSize));
        position_embeddings = nn.Parameter(zeros(1, positions, c.HiddenSize));
        patch_embeddings = new DinoPatchEmbeddings(c);
        RegisterComponents();
    }

    public override Tensor forward(Tensor pixelValues)
    {
        var height = pixelValues.shape[2];
        var width = pixelValues.shape[3];
        var embeddings = patch_embeddings.forward(pixelValues);
        embeddings = cat([cls_token.expand(embeddings.shape[0], -1, -1), embeddings], 1);
        return embeddings + PositionEncoding(embeddings, height, width);
    }

    private Tensor PositionEncoding(Tensor embeddings, long height, long width)
    {
        var numPatches = embeddings.shape[1] - 1;
        var numPositions = position_embeddings.shape[1] - 1;
        if (numPatches == numPositions && height == width) return position_embeddings;
        var dim = embeddings.shape[2];
        var cls = position_embeddings[.., ..1];
        var patch = position_embeddings[.., 1..];
        var side = (long)Math.Sqrt(numPositions);
        patch = patch.reshape(1, side, side, dim).permute(0, 3, 1, 2);
        patch = nn.functional.interpolate(patch, size: [height / _patch, width / _patch], mode: InterpolationMode.Bicubic, align_corners: false);
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim);
        return cat([cls, patch], 1);
    }
}

public sealed class DinoSelfAttention : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> query, key, value;
    private readonly int _heads;

    public DinoSelfAttention(DinoConfig c) : base(nameof(DinoSelfAttention))
    {
        _heads = c.NumAttentionHeads;
        query = nn.Linear(c.HiddenSize, c.HiddenSize);
        key = nn.Linear(c.HiddenSize, c.HiddenSize);
        value = nn.Linear(c.HiddenSize, c.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        long b = x.shape[0], n = x.shape[1], d = x.shape[2] / _heads;
        Tensor Split(Tensor t) => t.view(b, n, _heads, d).transpose(1, 2);
        using var q = Split(query.forward(x));
        using var k = Split(key.forward(x));
        using var v = Split(value.forward(x));
        using var scores = matmul(q, k.transpose(-1, -2)) * (1.0 / Math.Sqrt(d));
        using var probs = nn.functional.softmax(scores, -1);
        return matmul(probs, v).transpose(1, 2).contiguous().view(b, n, _heads * d);
    }
}

public sealed class DinoSelfOutput : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> dense;

    public DinoSelfOutput(DinoConfig c) : base(nameof(DinoSelfOutput))
    {
        dense = nn.Linear(c.HiddenSize, c.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => dense.forward(x);
}

public sealed class DinoAttention : nn.Module<Tensor, Tensor>
{
    private readonly DinoSelfAttention attention;
    private readonly DinoSelfOutput output;

    public DinoAttention(DinoConfig c) : base(nameof(DinoAttention))
    {
        attention = new DinoSelfAttention(c);
        output = new DinoSelfOutput(c);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) { using var a = attention.forward(x); return output.forward(a); }
}

public sealed class DinoIntermediate : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> dense;

    public DinoIntermediate(DinoConfig c) : base(nameof(DinoIntermediate))
    {
        dense = nn.Linear(c.HiddenSize, c.IntermediateSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) { using var h = dense.forward(x); return nn.functional.gelu(h); }
}

public sealed class DinoOutput : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> dense;

    public DinoOutput(DinoConfig c) : base(nameof(DinoOutput))
    {
        dense = nn.Linear(c.IntermediateSize, c.HiddenSize);
        RegisterComponents();
    }

    /// <summary>The second residual connection is made here, as in the reference.</summary>
    public override Tensor forward(Tensor x, Tensor residual) { using var h = dense.forward(x); return h + residual; }
}

public sealed class DinoLayer : nn.Module<Tensor, Tensor>
{
    private readonly DinoAttention attention;
    private readonly DinoIntermediate intermediate;
    private readonly DinoOutput output;
    private readonly nn.Module<Tensor, Tensor> layernorm_before, layernorm_after;

    public DinoLayer(DinoConfig c) : base(nameof(DinoLayer))
    {
        attention = new DinoAttention(c);
        intermediate = new DinoIntermediate(c);
        output = new DinoOutput(c);
        layernorm_before = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        layernorm_after = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden)
    {
        using var n1 = layernorm_before.forward(hidden);
        using var a = attention.forward(n1);
        using var h = a + hidden;
        using var n2 = layernorm_after.forward(h);
        using var i = intermediate.forward(n2);
        return output.forward(i, h);
    }
}

public sealed class DinoEncoder : nn.Module<Tensor, Tensor>
{
    private readonly ModuleList<DinoLayer> layer;

    public DinoEncoder(DinoConfig c) : base(nameof(DinoEncoder))
    {
        layer = nn.ModuleList(Enumerable.Range(0, c.NumHiddenLayers).Select(_ => new DinoLayer(c)).ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden)
    {
        var h = hidden;
        foreach (var l in layer)
        {
            var next = l.forward(h);
            if (!ReferenceEquals(h, hidden)) h.Dispose();
            h = next;
        }
        return h;
    }
}

/// <summary>The pooler of the Hugging Face model: the checkpoint holds its weights; TripoSR only uses the tokens, not the pooled output.</summary>
public sealed class DinoPooler : nn.Module
{
    private readonly nn.Module<Tensor, Tensor> dense;

    public DinoPooler(DinoConfig c) : base(nameof(DinoPooler))
    {
        dense = nn.Linear(c.HiddenSize, c.HiddenSize);
        RegisterComponents();
    }
}

/// <summary>
/// DINO ViT-B/16 as Hugging Face's <c>ViTModel</c> (<c>facebook/dino-vitb16</c>) with the position encoding interpolated: from the normalised picture
/// (B, 3, H, W) to the last hidden state (B, 1 + H/16 * W/16, 768). The names of its parameters are the checkpoint's <c>image_tokenizer.model.*</c> keys.
/// </summary>
public sealed class DinoViT : nn.Module<Tensor, Tensor>
{
    private readonly DinoEmbeddings embeddings;
    private readonly DinoEncoder encoder;
    private readonly nn.Module<Tensor, Tensor> layernorm;
    private readonly DinoPooler pooler;

    public DinoViT(DinoConfig? config = null) : base(nameof(DinoViT))
    {
        var c = config ?? new DinoConfig();
        embeddings = new DinoEmbeddings(c);
        encoder = new DinoEncoder(c);
        layernorm = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        pooler = new DinoPooler(c);
        RegisterComponents();
    }

    public override Tensor forward(Tensor pixelValues)
    {
        using var e = embeddings.forward(pixelValues);
        using var h = encoder.forward(e);
        return layernorm.forward(h);
    }
}

/// <summary>
/// <c>DINOSingleImageTokenizer</c>: the picture (B, 3, H, W) in 0..1 is normalised with the ImageNet mean and deviation and goes through the ViT.
/// Its only child is <c>model</c>, so the keys read <c>image_tokenizer.model.*</c>.
/// </summary>
public sealed class DinoSingleImageTokenizer : nn.Module<Tensor, Tensor>
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly DinoViT model;

    public DinoSingleImageTokenizer(DinoConfig? config = null) : base(nameof(DinoSingleImageTokenizer))
    {
        model = new DinoViT(config);
        RegisterComponents();
    }

    /// <summary>The image tokens (B, 1 + H/16 * W/16, 768): what <c>TSR.forward</c> feeds the backbone as <c>encoder_hidden_states</c> (the reference's channel-first layout, transposed).</summary>
    public override Tensor forward(Tensor images)
    {
        using var mean = tensor(Mean, images.dtype, images.device).reshape(1, 3, 1, 1);
        using var std = tensor(Std, images.dtype, images.device).reshape(1, 3, 1, 1);
        using var normalised = (images - mean) / std;
        return model.forward(normalised);
    }
}
