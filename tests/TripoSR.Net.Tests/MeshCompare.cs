using System.Buffers.Binary;
using System.Text.Json;
using TripoSR.Mesh;

namespace TripoSR.Net.Tests;

/// <summary>Comparison of two models as the spec asks: triangles, bounding box and a Chamfer distance, measured on the vertices.</summary>
internal static class MeshCompare
{
    /// <summary>The vertices (x y z) of the first primitive of a <c>.glb</c> and its triangle count.</summary>
    public static (float[] Vertices, int Triangles) ReadGlb(byte[] glb)
    {
        var jsonLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        using var doc = JsonDocument.Parse(glb.AsMemory(20, jsonLen));
        var root = doc.RootElement;
        var prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0];
        var accessors = root.GetProperty("accessors");
        var views = root.GetProperty("bufferViews");
        var pos = accessors[prim.GetProperty("attributes").GetProperty("POSITION").GetInt32()];
        var idx = accessors[prim.GetProperty("indices").GetInt32()];
        var binStart = 28 + jsonLen;
        var view = views[pos.GetProperty("bufferView").GetInt32()];
        var offset = binStart + view.GetProperty("byteOffset").GetInt32() + (pos.TryGetProperty("byteOffset", out var bo) ? bo.GetInt32() : 0);
        var count = pos.GetProperty("count").GetInt32();
        var v = new float[count * 3];
        for (var i = 0; i < v.Length; i++) v[i] = BinaryPrimitives.ReadSingleLittleEndian(glb.AsSpan(offset + i * 4));
        return (v, idx.GetProperty("count").GetInt32() / 3);
    }

    public static (double[] Min, double[] Max) Bounds(float[] v)
    {
        double[] lo = [double.MaxValue, double.MaxValue, double.MaxValue], hi = [double.MinValue, double.MinValue, double.MinValue];
        for (var i = 0; i < v.Length; i++)
        {
            lo[i % 3] = Math.Min(lo[i % 3], v[i]);
            hi[i % 3] = Math.Max(hi[i % 3], v[i]);
        }
        return (lo, hi);
    }

    /// <summary>The Chamfer distance (mean of the two mean nearest-vertex distances) divided by the diagonal of the reference's bounding box.</summary>
    public static double Chamfer(float[] a, float[] b)
    {
        var (lo, hi) = Bounds(b);
        var diagonal = Math.Sqrt(Enumerable.Range(0, 3).Sum(k => (hi[k] - lo[k]) * (hi[k] - lo[k])));
        return (MeanNearest(a, b) + MeanNearest(b, a)) / 2 / diagonal;
    }

    /// <summary>The mean distance from each vertex of <paramref name="from"/> to the nearest vertex of <paramref name="to"/>, with a grid of cells to look in.</summary>
    private static double MeanNearest(float[] from, float[] to)
    {
        var (lo, hi) = Bounds(to);
        var n = to.Length / 3;
        var cell = Math.Max(1e-6, Math.Cbrt((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) / Math.Max(1, n)) * 2);
        (int, int, int) Key(double x, double y, double z) => ((int)Math.Floor((x - lo[0]) / cell), (int)Math.Floor((y - lo[1]) / cell), (int)Math.Floor((z - lo[2]) / cell));
        var grid = new Dictionary<(int, int, int), List<int>>();
        for (var i = 0; i < n; i++)
        {
            var k = Key(to[i * 3], to[i * 3 + 1], to[i * 3 + 2]);
            if (!grid.TryGetValue(k, out var list)) grid[k] = list = [];
            list.Add(i);
        }
        double total = 0;
        for (var i = 0; i < from.Length / 3; i++)
        {
            double x = from[i * 3], y = from[i * 3 + 1], z = from[i * 3 + 2];
            var best = double.MaxValue;
            var (cx, cy, cz) = Key(x, y, z);
            for (var ring = 0; ring < 64; ring++)
            {
                for (var dx = -ring; dx <= ring; dx++)
                    for (var dy = -ring; dy <= ring; dy++)
                        for (var dz = -ring; dz <= ring; dz++)
                        {
                            if (Math.Max(Math.Abs(dx), Math.Max(Math.Abs(dy), Math.Abs(dz))) != ring) continue;
                            if (!grid.TryGetValue((cx + dx, cy + dy, cz + dz), out var list)) continue;
                            foreach (var j in list)
                            {
                                var d = (to[j * 3] - x) * (to[j * 3] - x) + (to[j * 3 + 1] - y) * (to[j * 3 + 1] - y) + (to[j * 3 + 2] - z) * (to[j * 3 + 2] - z);
                                if (d < best) best = d;
                            }
                        }
                // everything in the rings searched so far is within ring * cell; a point found closer than that cannot be beaten by the next ring
                if (best < double.MaxValue && Math.Sqrt(best) <= ring * cell) break;
            }
            total += Math.Sqrt(best);
        }
        return total / (from.Length / 3);
    }
}
