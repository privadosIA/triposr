using TorchSharp;
using static TorchSharp.torch;

namespace TripoSR.Field;

/// <summary>What the field says at a set of points: the density (after the bias and <c>exp</c>), the colour (after <c>sigmoid</c>), and the raw density before them.</summary>
public readonly record struct FieldSample(Tensor RawDensity, Tensor Density, Tensor Color) : IDisposable
{
    public void Dispose()
    {
        RawDensity.Dispose();
        Density.Dispose();
        Color.Dispose();
    }
}

/// <summary>
/// <c>TriplaneNeRFRenderer.query_triplane</c> (the only part of the renderer that meshing uses): the points (inside the cube of side 2 * radius) are
/// projected on the three planes of the scene codes, sampled with <c>grid_sample</c> (bilinear, zero outside), the three results are concatenated and the
/// <see cref="NeRFMLP"/> answers; the density is <c>exp(raw - 1)</c> and the colour <c>sigmoid</c> of the features. Points go through in chunks.
/// </summary>
public static class TriplaneField
{
    /// <summary>The chunk of points that goes through the MLP at once (the container sets 8192).</summary>
    public const int DefaultChunk = 8192;

    /// <param name="decoder">The MLP.</param>
    /// <param name="positions">(N, 3), in (-radius, radius).</param>
    /// <param name="sceneCode">One scene code: (3, 40, 64, 64).</param>
    public static FieldSample Query(NeRFMLP decoder, Tensor positions, Tensor sceneCode, TripoSRConfig? config = null, int chunk = DefaultChunk, bool withColor = true)
    {
        var c = config ?? new TripoSRConfig();
        var n = positions.shape[0];
        using var scaled = ScaleTensor(positions, -c.Radius, c.Radius, -1.0, 1.0); // to (-1, 1) for grid_sample
        var raw = new List<Tensor>();
        var features = new List<Tensor>();
        for (long start = 0; start < Math.Max(1, n); start += chunk)
        {
            using var scope = NewDisposeScope(); // the temporaries of a chunk go when it is done (on a card they would pile up until the garbage collector runs)
            using var x = scaled.narrow(0, start, Math.Min(chunk, n - start));
            var planes = stack([x.index_select(-1, tensor(new long[] { 0, 1 }, device: x.device)),
                                x.index_select(-1, tensor(new long[] { 0, 2 }, device: x.device)),
                                x.index_select(-1, tensor(new long[] { 1, 2 }, device: x.device))], -3); // (3, N, 2)
            var grid = planes.unsqueeze(1);                                                               // (3, 1, N, 2)
            var sampled = nn.functional.grid_sample(sceneCode, grid, GridSampleMode.Bilinear, GridSamplePaddingMode.Zeros, false); // (3, C, 1, N)
            var cat = sampled.squeeze(2).permute(2, 0, 1).reshape(x.shape[0], -1);                        // (N, 3 * C)
            using var o = decoder.forward(cat);
            raw.Add(o.Density.clone().MoveToOuterDisposeScope());
            if (withColor) features.Add(o.Features.clone().MoveToOuterDisposeScope());
        }
        using var rawAll = raw.Count == 1 ? raw[0].alias() : cat(raw, 0);
        using var featAll = !withColor ? null : features.Count == 1 ? features[0].alias() : cat(features, 0);
        foreach (var t in raw.Concat(features)) t.Dispose();
        using var biased = rawAll + c.DensityBias;
        return new FieldSample(rawAll.clone(), exp(biased), featAll is null ? empty([0, 3]) : sigmoid(featAll));
    }

    /// <summary>
    /// Only the density (after the bias and <c>exp</c>), point by point in chunks into one preallocated tensor: what the isosurface needs for a grid of millions of points
    /// without keeping the colour features of all of them.
    /// </summary>
    /// <param name="positions">(N, 3), in (-radius, radius).</param>
    /// <returns>(N, 1) float32.</returns>
    public static Tensor QueryDensity(NeRFMLP decoder, Tensor positions, Tensor sceneCode, TripoSRConfig? config = null, int chunk = DefaultChunk)
    {
        var c = config ?? new TripoSRConfig();
        var n = positions.shape[0];
        var result = empty([n, 1], ScalarType.Float32, positions.device);
        for (long start = 0; start < n; start += chunk)
        {
            using var scope = NewDisposeScope();
            var len = Math.Min(chunk, n - start);
            var x = positions.narrow(0, start, len);
            var s = Query(decoder, x, sceneCode, c, chunk: (int)len, withColor: false);
            result.narrow(0, start, len).copy_(s.Density);
        }
        return result;
    }

    /// <summary><c>tsr.utils.scale_tensor</c>: <c>(x - src0) / (src1 - src0) * (tgt1 - tgt0) + tgt0</c>, the scalars as Python doubles.</summary>
    public static Tensor ScaleTensor(Tensor x, double src0, double src1, double tgt0, double tgt1)
    {
        using var a = x - src0;
        using var b = a / (src1 - src0);
        using var c = b * (tgt1 - tgt0);
        return c + tgt0;
    }
}
