using System.Buffers.Binary;
using System.Text.Json;
using TripoSR.Mesh;
using static TorchSharp.torch;

namespace TripoSR.Net.Tests;

/// <summary>The finish of the container (6.2) and the <c>.glb</c> (6.3): upright, grounded, the triangle limit with its retreat to a lower detail, and a file the orchestrator reads.</summary>
public sealed class ModelFinishTests
{
    private static readonly string[] Names = ["chair", "hamburger", "horse", "teapot", "poly_fox"];
    private static readonly (string Detail, int Size)[] Details = [("low", 128), ("normal", 192), ("high", 256)];

    public static IEnumerable<object[]> Cases() => Names.SelectMany(n => Details.Select(d => new object[] { n, d.Detail, d.Size }));

    /// <summary>The mesh of the reference as extract_mesh gave it (after trimesh merged its vertices), from the golden run.</summary>
    private static ColoredMesh RawMesh(string name, int size)
    {
        var t = Fixtures.Load($"{name}.{size}.mesh.safetensors");
        using var v = t["vertices"];
        using var f = t["faces"];
        using var c = t["colors"];
        var colors = c.data<float>().ToArray().Select(x => (byte)Math.Round(x * 255f)).ToArray();
        return new ColoredMesh(v.data<float>().ToArray(), f.data<long>().ToArray().Select(i => (int)i).ToArray(), colors);
    }

    private static ColoredMesh Box(int triangles) =>
        new([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], Enumerable.Range(0, triangles * 3).Select(i => i % 4).ToArray(), new byte[16]);

    private sealed record Reference(int Triangles, double[] Min, double[] Max, string DetailUsed);

    private static Reference ReferenceOf(string name, string detail)
    {
        using var doc = JsonDocument.Parse(Fixtures.ReadAllText(name + ".json"));
        var m = doc.RootElement.GetProperty("models").GetProperty(detail);
        double[] Arr(string k) => m.GetProperty(k).EnumerateArray().Select(e => e.GetDouble()).ToArray();
        return new Reference(m.GetProperty("triangles").GetInt32(), Arr("bounds_min"), Arr("bounds_max"), m.GetProperty("detail_used").GetString()!);
    }

    [Theory, MemberData(nameof(Cases))]
    public void TheFinishedModelStandsOnItsBaseAsInTheContainer(string name, string detail, int size)
    {
        var done = ModelFinish.UprightAndGrounded(RawMesh(name, size));
        var r = ReferenceOf(name, detail);
        Assert.Equal(r.Triangles, done.TriangleCount);
        double[] lo = [double.MaxValue, double.MaxValue, double.MaxValue], hi = [double.MinValue, double.MinValue, double.MinValue];
        for (var i = 0; i < done.Vertices.Length; i++)
        {
            lo[i % 3] = Math.Min(lo[i % 3], done.Vertices[i]);
            hi[i % 3] = Math.Max(hi[i % 3], done.Vertices[i]);
        }
        for (var k = 0; k < 3; k++)
        {
            Assert.Equal(r.Min[k], lo[k], 6);
            Assert.Equal(r.Max[k], hi[k], 6);
        }
        Assert.Equal(0.0, lo[1]);                          // the base is at Y = 0
        Assert.Equal(0.0, lo[0] + hi[0], 6);               // centred on X and Z
        Assert.Equal(0.0, lo[2] + hi[2], 6);
    }

    [Fact]
    public void TheTurnSendsXYZToMinusYZMinusX()
    {
        var m = new ColoredMesh([1, 2, 3, 4, 5, 6, 0, 0, 0], [0, 1, 2], new byte[12]);
        var o = ModelFinish.UprightAndGrounded(m);
        // (x, y, z) -> (-y, z, -x): (-2, 3, -1), (-5, 6, -4), (0, 0, 0); then lowest Y to 0 and the middle of X and Z to 0
        Assert.Equal([-2f + 2.5f, 3f, -1f + 2f, -5f + 2.5f, 6f, -4f + 2f, 0f + 2.5f, 0f, 0f + 2f], o.Vertices);
    }

    [Theory, MemberData(nameof(Cases))]
    public void TheGlbHasTheBytesOfTheModelsOfTheContainer(string name, string detail, int size)
    {
        var glb = GlbWriter.Write(ModelFinish.UprightAndGrounded(RawMesh(name, size)));
        var expected = File.ReadAllBytes(Fixtures.PathOf($"{name}.{detail}.glb"));
        Assert.Equal(Bin(expected), Bin(glb));   // indices, positions and colours, byte for byte
        Assert.Equal(Inspect(expected).Triangles, Inspect(glb).Triangles);
    }

    private static byte[] Bin(byte[] glb)
    {
        var jsonLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        var binLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(20 + jsonLen));
        return glb.AsSpan(28 + jsonLen, binLen).ToArray();
    }

    /// <summary>What <c>blender.ParseModel</c> of Veritmo checks and counts: the header, the JSON chunk, no files outside, and the triangles from the accessors.</summary>
    private static (int Triangles, int Bytes) Inspect(byte[] data)
    {
        Assert.True(data.Length >= 20 && data.Length <= 32 << 20, "too short or larger than 32 MiB");
        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(data));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(4)));
        Assert.True(BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(8)) <= data.Length, "cut short");
        var size = (int)BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(12));
        Assert.Equal(0x4E4F534Au, BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(16)));
        Assert.True(size <= data.Length - 20, "the JSON chunk is cut short");
        using var doc = JsonDocument.Parse(data.AsMemory(20, size));
        var root = doc.RootElement;
        foreach (var b in root.GetProperty("buffers").EnumerateArray())
            Assert.False(b.TryGetProperty("uri", out var uri) && uri.GetString() is { Length: > 0 } u && !u.StartsWith("data:"), "the data is in another file");
        var accessors = root.GetProperty("accessors").EnumerateArray().Select(a => a.GetProperty("count").GetInt32()).ToArray();
        var triangles = 0;
        foreach (var mesh in root.GetProperty("meshes").EnumerateArray())
            foreach (var p in mesh.GetProperty("primitives").EnumerateArray())
            {
                var mode = p.TryGetProperty("mode", out var md) ? md.GetInt32() : 4;
                var n = p.TryGetProperty("indices", out var ix) ? accessors[ix.GetInt32()] : accessors[p.GetProperty("attributes").GetProperty("POSITION").GetInt32()];
                if (mode == 4) triangles += n / 3;
                else if ((mode == 5 || mode == 6) && n > 2) triangles += n - 2;
            }
        return (triangles, data.Length);
    }

    [Theory, MemberData(nameof(Cases))]
    public void TheOrchestratorReadsTheFile(string name, string detail, int size)
    {
        var mesh = ModelFinish.UprightAndGrounded(RawMesh(name, size));
        var (triangles, bytes) = Inspect(GlbWriter.Write(mesh));
        Assert.Equal(mesh.TriangleCount, triangles);
        Assert.InRange(triangles, 1, EngineInfo.MaxTriangles);
        Assert.True(bytes > 0);
    }

    [Fact]
    public void ADetailTooHighForASceneFallsBackToTheNextLower()
    {
        var asked = new List<int>();
        var made = ModelFinish.Make("high", size => { asked.Add(size); return Box(size == 256 ? 900 : 100); }, maxTriangles: 500);
        Assert.Equal([256, 192], asked);
        Assert.Equal(("normal", 192, 100), (made.DetailUsed, made.Size, made.Mesh.TriangleCount));
    }

    [Fact]
    public void WhenEvenTheLowestHasTooManyTrianglesItIsRefusedWithASentence()
    {
        var e = Assert.Throws<ModelRefusedException>(() => ModelFinish.Make("normal", _ => Box(700), maxTriangles: 500));
        Assert.Equal("700 triangles, more than the 500 of a scene", e.Message);
    }

    [Fact]
    public void AnEmptyOrSurfacelessModelIsRefused()
    {
        Assert.Contains("no surface", Assert.Throws<ModelRefusedException>(() => ModelFinish.Make("low", _ => new ColoredMesh([], [], []))).Message);
        Assert.Contains("no surface", Assert.Throws<ModelRefusedException>(() => ModelFinish.Make("low", _ => throw new NoSurfaceException("x"))).Message);
    }

    [Fact]
    public void ADetailThatDoesNotExistIsRefused() =>
        Assert.Contains("low, normal or high", Assert.Throws<ModelRefusedException>(() => ModelFinish.DetailsFrom("ultra")).Message);
}
