using TorchSharp;
using static TorchSharp.torch;

namespace TripoSR.Field;

#pragma warning disable IDE1006 // naming: the names are the checkpoint's keys (decoder.layers.N.*)

/// <summary>What the field answers at a point: the raw density (before the bias and the exponential) and the raw colour features.</summary>
public readonly record struct FieldOutput(Tensor Density, Tensor Features) : IDisposable
{
    public void Dispose()
    {
        Density.Dispose();
        Features.Dispose();
    }
}

/// <summary>
/// <c>NeRFMLP</c> (the decoder): 120 features (3 planes x 40 channels) through 9 hidden layers of 64 with SiLU, then 4 outputs: the density and 3 colour features.
/// The layers are numbered as in the reference's <c>nn.Sequential</c> (a linear every other index, the activations between), so the keys are <c>decoder.layers.0</c> ... <c>decoder.layers.18</c>.
/// </summary>
public sealed class NeRFMLP : nn.Module<Tensor, FieldOutput>
{
    private readonly nn.Module<Tensor, Tensor> layers;

    public NeRFMLP(TripoSRConfig? config = null) : base(nameof(NeRFMLP))
    {
        var c = config ?? new TripoSRConfig();
        var list = new List<nn.Module<Tensor, Tensor>> { nn.Linear(3 * c.SceneChannels, c.DecoderHidden), nn.SiLU() };
        for (var i = 0; i < c.DecoderHiddenLayers - 1; i++)
        {
            list.Add(nn.Linear(c.DecoderHidden, c.DecoderHidden));
            list.Add(nn.SiLU());
        }
        list.Add(nn.Linear(c.DecoderHidden, 4)); // density 1 + features 3
        layers = nn.Sequential(list.Select((m, i) => (i.ToString(), m)));
        RegisterComponents();
    }

    /// <summary>(..., 120) to the density (..., 1) and the features (..., 3).</summary>
    public override FieldOutput forward(Tensor x)
    {
        var shape = x.shape[..^1];
        using var flat = x.reshape(-1, x.shape[^1]);
        using var f = layers.forward(flat);
        using var all = f.reshape([.. shape, -1]);
        return new FieldOutput(all.narrow(-1, 0, 1).contiguous(), all.narrow(-1, 1, 3).contiguous());
    }
}
