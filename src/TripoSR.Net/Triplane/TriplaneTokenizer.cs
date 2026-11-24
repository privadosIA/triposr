using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TripoSR.Triplane;

#pragma warning disable IDE1006 // naming: the names are the checkpoint's keys (tokenizer.embeddings)

/// <summary>
/// <c>Triplane1DTokenizer</c>: the learned embedding of the three planes (3 x 1024 x 32 x 32) as a sequence of tokens, and back.
/// The tokens are channel-first, (B, 1024, 3 * 32 * 32), as the reference has them.
/// </summary>
public sealed class TriplaneTokenizer : nn.Module
{
    private readonly Parameter embeddings;
    private readonly int _plane, _channels;

    public TriplaneTokenizer(TripoSRConfig? config = null) : base(nameof(TriplaneTokenizer))
    {
        var c = config ?? new TripoSRConfig();
        (_plane, _channels) = (c.PlaneSize, c.TokenChannels);
        embeddings = nn.Parameter(zeros(3, c.TokenChannels, c.PlaneSize, c.PlaneSize));
        RegisterComponents();
    }

    /// <summary>(B, C, 3 * H * W): <c>rearrange(repeat(embeddings), "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")</c>.</summary>
    public Tensor forward(int batchSize) =>
        embeddings.unsqueeze(0).expand(batchSize, -1, -1, -1, -1).permute(0, 2, 1, 3, 4).reshape(batchSize, _channels, 3 * _plane * _plane);

    /// <summary>(B, C, 3 * H * W) to (B, 3, C, H, W).</summary>
    public Tensor detokenize(Tensor tokens)
    {
        long b = tokens.shape[0], ct = tokens.shape[1], nt = tokens.shape[2];
        if (nt != 3L * _plane * _plane) throw new ArgumentException($"{nt} tokens, {3 * _plane * _plane} expected");
        if (ct != _channels) throw new ArgumentException($"{ct} channels, {_channels} expected");
        return tokens.reshape(b, ct, 3, _plane, _plane).permute(0, 2, 1, 3, 4).contiguous();
    }
}
