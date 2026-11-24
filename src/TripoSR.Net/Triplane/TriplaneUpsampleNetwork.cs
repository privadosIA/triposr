using TorchSharp;
using static TorchSharp.torch;

namespace TripoSR.Triplane;

#pragma warning disable IDE1006 // naming: the names are the checkpoint's keys (post_processor.upsample.*)

/// <summary><c>TriplaneUpsampleNetwork</c>: a transposed convolution (kernel 2, stride 2) from 1024 to 40 channels on each of the three planes, 32x32 to 64x64.</summary>
public sealed class TriplaneUpsampleNetwork : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> upsample;

    public TriplaneUpsampleNetwork(TripoSRConfig? config = null) : base(nameof(TriplaneUpsampleNetwork))
    {
        var c = config ?? new TripoSRConfig();
        upsample = nn.ConvTranspose2d(c.TokenChannels, c.SceneChannels, 2, stride: 2);
        RegisterComponents();
    }

    /// <summary>(B, 3, 1024, 32, 32) to the scene codes (B, 3, 40, 64, 64).</summary>
    public override Tensor forward(Tensor triplanes)
    {
        long b = triplanes.shape[0];
        using var flat = triplanes.reshape(b * 3, triplanes.shape[2], triplanes.shape[3], triplanes.shape[4]);
        using var up = upsample.forward(flat);
        return up.reshape(b, 3, up.shape[1], up.shape[2], up.shape[3]);
    }
}
