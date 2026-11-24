using TorchSharp;
using static TorchSharp.torch;

namespace TripoSR.Transformer;

#pragma warning disable IDE1006 // naming: the names are the checkpoint's keys (backbone.*)

/// <summary>An attention layer of the reference (<c>Attention</c> with <c>AttnProcessor2_0</c>): no bias on the queries, keys and values, one on the output.</summary>
public sealed class Attention : nn.Module<Tensor, Tensor?, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> to_q, to_k, to_v;
    private readonly nn.Module<Tensor, Tensor> to_out;
    private readonly int _heads;

    public Attention(int queryDim, int heads, int headDim, int? crossAttentionDim) : base(nameof(Attention))
    {
        _heads = heads;
        var inner = heads * headDim;
        var kv = crossAttentionDim ?? queryDim;
        to_q = nn.Linear(queryDim, inner, hasBias: false);
        to_k = nn.Linear(kv, inner, hasBias: false);
        to_v = nn.Linear(kv, inner, hasBias: false);
        to_out = nn.Sequential(("0", nn.Linear(inner, queryDim))); // a ModuleList of the linear and a dropout in the reference
        RegisterComponents();
    }

    /// <summary>Self-attention when <paramref name="context"/> is null, otherwise attention to it.</summary>
    public override Tensor forward(Tensor x, Tensor? context)
    {
        var c = context ?? x;
        long b = x.shape[0];
        using var q0 = to_q.forward(x);
        using var k0 = to_k.forward(c);
        using var v0 = to_v.forward(c);
        var d = q0.shape[2] / _heads;
        using var q = q0.view(b, -1, _heads, d).transpose(1, 2);
        using var k = k0.view(b, -1, _heads, d).transpose(1, 2);
        using var v = v0.view(b, -1, _heads, d).transpose(1, 2);
        using var o = nn.functional.scaled_dot_product_attention(q, k, v);
        using var merged = o.transpose(1, 2).reshape(b, -1, _heads * d);
        return to_out.forward(merged);
    }
}

/// <summary><c>GEGLU</c>: a linear to twice the width, one half times the GELU of the other.</summary>
public sealed class GEGLU : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> proj;

    public GEGLU(int dimIn, int dimOut) : base(nameof(GEGLU))
    {
        proj = nn.Linear(dimIn, dimOut * 2);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var p = proj.forward(x);
        var halves = p.chunk(2, -1);
        using var gate = nn.functional.gelu(halves[1]);
        var o = halves[0] * gate;
        foreach (var h in halves) h.Dispose();
        return o;
    }
}

/// <summary>The feed-forward: GEGLU to four times the width, then a linear back (the dropout between them has no parameters and does nothing in inference).</summary>
public sealed class FeedForward : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> net;

    public FeedForward(int dim) : base(nameof(FeedForward))
    {
        net = nn.Sequential(("0", new GEGLU(dim, dim * 4)), ("1", nn.Identity()), ("2", nn.Linear(dim * 4, dim)));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => net.forward(x);
}

/// <summary>Self-attention, cross-attention to the image tokens and the feed-forward, each after a LayerNorm and with a residual connection.</summary>
public sealed class BasicTransformerBlock : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> norm1, norm2, norm3;
    private readonly Attention attn1, attn2;
    private readonly FeedForward ff;

    public BasicTransformerBlock(int dim, int heads, int headDim, int crossAttentionDim) : base(nameof(BasicTransformerBlock))
    {
        norm1 = nn.LayerNorm(dim);
        attn1 = new Attention(dim, heads, headDim, null);
        norm2 = nn.LayerNorm(dim);
        attn2 = new Attention(dim, heads, headDim, crossAttentionDim);
        norm3 = nn.LayerNorm(dim);
        ff = new FeedForward(dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden, Tensor encoderHidden)
    {
        using var n1 = norm1.forward(hidden);
        using var a1 = attn1.forward(n1, null);
        using var h1 = a1 + hidden;
        using var n2 = norm2.forward(h1);
        using var a2 = attn2.forward(n2, encoderHidden);
        using var h2 = a2 + h1;
        using var n3 = norm3.forward(h2);
        using var f = ff.forward(n3);
        return f + h2;
    }
}

/// <summary>
/// <c>Transformer1D</c> (the backbone): the tokens (B, 1024, N) are normalised by groups, projected, go through 16 blocks that attend to the image tokens
/// (B, Nt, 768) and are projected back, plus the input (the residual).
/// </summary>
public sealed class Transformer1D : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> norm;
    private readonly nn.Module<Tensor, Tensor> proj_in, proj_out;
    private readonly TorchSharp.Modules.ModuleList<BasicTransformerBlock> transformer_blocks;

    public Transformer1D(TripoSRConfig? config = null) : base(nameof(Transformer1D))
    {
        var c = config ?? new TripoSRConfig();
        norm = nn.GroupNorm(c.NormGroups, c.TokenChannels, 1e-6);
        proj_in = nn.Linear(c.TokenChannels, c.InnerDim);
        transformer_blocks = nn.ModuleList(Enumerable.Range(0, c.Layers).Select(_ => new BasicTransformerBlock(c.InnerDim, c.Heads, c.HeadDim, c.CrossAttentionDim)).ToArray());
        proj_out = nn.Linear(c.InnerDim, c.TokenChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor tokens, Tensor encoderHiddenStates)
    {
        using var normed = norm.forward(tokens);                 // (B, C, N)
        using var seq = normed.permute(0, 2, 1).contiguous();    // (B, N, C)
        var h = proj_in.forward(seq);
        foreach (var block in transformer_blocks)
        {
            var next = block.forward(h, encoderHiddenStates);
            h.Dispose();
            h = next;
        }
        using var projected = proj_out.forward(h);
        h.Dispose();
        using var back = projected.permute(0, 2, 1);
        return back + tokens;
    }
}
