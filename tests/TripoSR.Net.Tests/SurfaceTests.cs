using MusicGen.Core.Testing;
using TorchSharp;
using TripoSR.Field;
using TripoSR.Mesh;
using TripoSR.Weights;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>The surface (6.1): the density grid, marching cubes in the reference's axes and winding, and the colour per vertex, against the golden run.</summary>
public sealed class SurfaceTests(ITestOutputHelper output)
{
    private static readonly Lazy<NeRFMLP> Decoder = new(() =>
    {
        var m = new NeRFMLP();
        TorchCheckpointLoader.Load(m, Checkpoint.Model, CPU, "decoder.", "The checkpoint does not match the decoder:", "decoder keys", "decoder.");
        m.eval();
        return m;
    });

    public static IEnumerable<object[]> Pictures() => new[] { "chair", "hamburger", "horse", "teapot", "poly_fox" }.Select(n => new object[] { n });
    public static IEnumerable<object[]> PicturesAndResolutions() => Pictures().SelectMany(p => new[] { 128, 192, 256 }.Select(r => new[] { p[0], (object)r }));

    [Fact]
    public void AnEmptyVolumeHasNoSurface()
    {
        var e = Assert.Throws<NoSurfaceException>(() => MarchingCubes.Run(new float[27], 3, 3, 3, 0.5));
        Assert.Contains("level", e.Message);
    }

    [Fact]
    public void ASphereIsAClosedSurfaceOfTheRightSize()
    {
        const int n = 40;
        var volume = new float[n * n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                for (var k = 0; k < n; k++)
                    volume[(i * n + j) * n + k] = 12f - MathF.Sqrt((i - 19.5f) * (i - 19.5f) + (j - 19.5f) * (j - 19.5f) + (k - 19.5f) * (k - 19.5f)); // inside > 0
        var mesh = MarchingCubes.Run(volume, n, n, n, 0.0);
        // every vertex is at radius 12 from the centre (within a voxel), and every edge belongs to two triangles: a closed surface
        for (var i = 0; i < mesh.VertexCount; i++)
        {
            var r = MathF.Sqrt(MathF.Pow(mesh.Vertices[i * 3] - 19.5f, 2) + MathF.Pow(mesh.Vertices[i * 3 + 1] - 19.5f, 2) + MathF.Pow(mesh.Vertices[i * 3 + 2] - 19.5f, 2));
            Assert.InRange(r, 11.0f, 13.0f);
        }
        var edges = new Dictionary<(int, int), int>();
        for (var t = 0; t < mesh.TriangleCount; t++)
            for (var e = 0; e < 3; e++)
            {
                int a = mesh.Faces[t * 3 + e], b = mesh.Faces[t * 3 + (e + 1) % 3];
                var key = a < b ? (a, b) : (b, a);
                edges[key] = edges.GetValueOrDefault(key) + 1;
            }
        Assert.All(edges.Values, c => Assert.Equal(2, c));
    }

    [RequiresCheckpointTheory, MemberData(nameof(PicturesAndResolutions))]
    public void TheSurfaceIsTheOneOfTheReference(string name, int resolution)
    {
        var stages = Fixtures.Load(name + ".stages.safetensors");
        using var codes = stages["scene_codes"];
        using var sceneCode = codes[0];
        var expected = Fixtures.Load($"{name}.{resolution}.mesh.safetensors");
        using var ev = expected["vertices"];
        using var ef = expected["faces"];
        using var ec = expected["colors"];

        var mesh = SurfaceExtractor.Extract(Decoder.Value, sceneCode, resolution);
        output.WriteLine($"{name} {resolution}: {mesh.VertexCount} vertices, {mesh.TriangleCount} triangles (reference {ev.shape[0]}, {ef.shape[0]})");
        Assert.Equal(ef.shape[0], mesh.TriangleCount);
        Assert.Equal(ev.shape[0], mesh.VertexCount);

        // the faces in the same order, and each one with the same corners (the order of the vertices is trimesh's hash order when it merged some)
        var faces = ef.data<long>().ToArray();
        var rv = ev.data<float>().ToArray();
        var rc = ec.data<float>().ToArray();
        float maxPos = 0, maxCol = 0;
        for (var i = 0; i < faces.Length; i++)
        {
            int g = mesh.Faces[i], e = (int)faces[i];
            for (var k = 0; k < 3; k++) maxPos = Math.Max(maxPos, Math.Abs(mesh.Vertices[g * 3 + k] - rv[e * 3 + k]));
            for (var k = 0; k < 4; k++) maxCol = Math.Max(maxCol, Math.Abs(mesh.Colors[g * 4 + k] / 255f - rc[e * 4 + k]));
        }
        output.WriteLine($"  corners of the faces: max position difference {maxPos:E2}, max colour difference {maxCol * 255:F2} levels");
        Assert.True(maxPos <= 1e-4, $"max position difference {maxPos}");
        Assert.True(maxCol <= 1.0 / 255 + 1e-6, $"max colour difference {maxCol * 255} levels");
    }
}
