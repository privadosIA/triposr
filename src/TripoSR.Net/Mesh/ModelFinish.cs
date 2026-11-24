namespace TripoSR.Mesh;

/// <summary>A request the engine will not make, with the sentence the caller is told (the <c>Refused</c> of the container's <c>server.py</c>).</summary>
public sealed class ModelRefusedException(string message) : Exception(message);

/// <summary>A model over the triangles a scene takes: the caller makes it again with less detail.</summary>
public sealed class TooManyTrianglesException(int triangles, int max) : Exception($"{triangles} triangles, more than the {max} of a scene")
{
    public int Triangles { get; } = triangles;
}

/// <summary>A finished model and the detail it was made at.</summary>
public sealed record FinishedModel(ColoredMesh Mesh, string DetailUsed, int Size);

/// <summary>
/// What every model leaves with, whatever the engine gave (the contract of the container's <c>server.py</c>): upright in glTF's axes (Y up), its origin at the middle of
/// its base, with surface and with no more triangles than a scene takes, its colours kept. A model with too many triangles is not decimated: it is made again at the next
/// lower detail (the container never decimates), and the detail it ended at is reported.
/// </summary>
public static class ModelFinish
{
    /// <summary>The details, highest first.</summary>
    public static readonly IReadOnlyList<string> DetailOrder = ["high", "normal", "low"];

    /// <summary>The details to try, from the one asked for down.</summary>
    /// <exception cref="ModelRefusedException">The detail is none of the three.</exception>
    public static IReadOnlyList<string> DetailsFrom(string detail)
    {
        var i = DetailOrder.ToList().IndexOf(detail);
        if (i < 0) throw new ModelRefusedException($"detail must be low, normal or high, got '{detail}'");
        return DetailOrder.Skip(i).ToList();
    }

    /// <summary>
    /// TripoSR's meshes are not upright in glTF's Y-up (its own demo turns them -90 degrees about X and then 90 about Y): turn them so, then put the origin at the middle of the base.
    /// The two turns send <c>(x, y, z)</c> to <c>(-y, z, -x)</c>; they are made as trimesh makes them (rotation matrices in double precision, whose residuals of 1e-16 decide which way a
    /// rounding tie goes when the result is stored as float32), so the vertices are the reference's to the last bit.
    /// </summary>
    public static ColoredMesh UprightAndGrounded(ColoredMesh mesh)
    {
        var v = mesh.Vertices.Select(x => (double)x).ToArray();
        v = Transform(v, RotationMatrix(-Math.PI / 2, [1, 0, 0]));
        v = Transform(v, RotationMatrix(Math.PI / 2, [0, 1, 0]));
        return Grounded(v, mesh);
    }

    /// <summary><c>trimesh.transformations.rotation_matrix(angle, direction)</c> (4x4, about the origin), term by term.</summary>
    private static double[,] RotationMatrix(double angle, double[] direction)
    {
        double sina = Math.Sin(angle), cosa = Math.Cos(angle);
        var norm = Math.Sqrt(direction.Sum(d => d * d));
        var d = direction.Select(x => x / norm).ToArray();
        var m = new double[4, 4];
        for (var i = 0; i < 3; i++) m[i, i] = cosa;
        m[3, 3] = 1.0;
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++) m[i, j] += d[i] * d[j] * (1.0 - cosa);
        var s = d.Select(x => x * sina).ToArray();
        double[,] skew = { { 0.0, -s[2], s[1] }, { s[2], 0.0, -s[0] }, { -s[1], s[0], 0.0 } };
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++) m[i, j] += skew[i, j];
        return m;
    }

    /// <summary><c>transform_points</c>: <c>matrix @ [x, y, z, 1]</c> for every point.</summary>
    private static double[] Transform(double[] points, double[,] m)
    {
        var o = new double[points.Length];
        for (var i = 0; i < points.Length; i += 3)
            for (var r = 0; r < 3; r++)
                o[i + r] = m[r, 0] * points[i] + m[r, 1] * points[i + 1] + m[r, 2] * points[i + 2] + m[r, 3] * 1.0;
        return o;
    }

    /// <summary>The origin at the middle of the base: the lowest point of the bounding box (Y), centred on the other two axes.</summary>
    public static ColoredMesh Grounded(ColoredMesh mesh) => Grounded(mesh.Vertices.Select(v => (double)v).ToArray(), mesh);

    private static ColoredMesh Grounded(double[] v, ColoredMesh like)
    {
        double[] lo = [double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity], hi = [double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity];
        for (var i = 0; i < v.Length; i++)
        {
            var k = i % 3;
            if (v[i] < lo[k]) lo[k] = v[i];
            if (v[i] > hi[k]) hi[k] = v[i];
        }
        var shift = new[] { -(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2 };
        var o = new float[v.Length];
        for (var i = 0; i < v.Length; i++) o[i] = (float)(v[i] + shift[i % 3]);
        return like with { Vertices = o };
    }

    /// <summary>A model with no surface is refused; one with more triangles than a scene takes is <see cref="TooManyTrianglesException"/>.</summary>
    /// <returns>The triangles.</returns>
    public static int Check(ColoredMesh mesh, int maxTriangles = EngineInfo.MaxTriangles)
    {
        var n = mesh.TriangleCount;
        if (n == 0 || mesh.VertexCount == 0) throw new ModelRefusedException("the engine made no surface: the picture may not show an object on its own");
        if (n > maxTriangles) throw new TooManyTrianglesException(n, maxTriangles);
        return n;
    }

    /// <summary>
    /// Makes the model at the detail asked, finished, and at the next lower ones while it has too many triangles (<paramref name="extract"/> gives the mesh of a resolution).
    /// </summary>
    /// <exception cref="ModelRefusedException">No surface, or too many triangles even at the lowest detail.</exception>
    public static FinishedModel Make(string detail, Func<int, ColoredMesh> extract, int maxTriangles = EngineInfo.MaxTriangles)
    {
        TooManyTrianglesException? last = null;
        foreach (var d in DetailsFrom(detail))
        {
            var size = EngineInfo.DetailSizes[d];
            ColoredMesh mesh;
            try { mesh = UprightAndGrounded(extract(size)); }
            catch (NoSurfaceException) { throw new ModelRefusedException("the engine made no surface: the picture may not show an object on its own"); }
            try
            {
                Check(mesh, maxTriangles);
                return new FinishedModel(mesh, d, size);
            }
            catch (TooManyTrianglesException e) { last = e; }
        }
        throw new ModelRefusedException(last!.Message);
    }
}
