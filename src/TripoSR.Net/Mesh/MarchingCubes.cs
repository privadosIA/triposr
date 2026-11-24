// A port of skimage.measure._marching_cubes_lewiner_cy (scikit-image 0.26.0, Cython by Almar Klein after Thomas Lewiner's C++), the marching cubes that
// the reference mesh was made with: the same cells in the same order, the same cases and the same vertex sharing, so that the same volume gives the same
// vertices and triangles. Only what the mesh needs is ported (no normals or per-vertex values).
//
// Copyright (C) 2009-2022 the scikit-image team. BSD-3-Clause (https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt).
namespace TripoSR.Mesh;

/// <summary>The volume has no surface at the level asked for (the level is out of its range, or nothing crosses it).</summary>
public sealed class NoSurfaceException(string message) : InvalidOperationException(message);

/// <summary>The vertices (x y z as consecutive floats) and the triangles (three vertex indices each) of a surface.</summary>
public sealed record SurfaceMesh(float[] Vertices, int[] Faces)
{
    public int VertexCount => Vertices.Length / 3;
    public int TriangleCount => Faces.Length / 3;
}

/// <summary>
/// Marching cubes of Lewiner et al. (2003) on a volume of floats, with topological guarantees. The result is in the order the reference's <c>MarchingCubeHelper</c> leaves
/// it: vertices as <c>(axis 0, axis 1, axis 2)</c> coordinates in voxel units (what <c>skimage.measure.marching_cubes</c> returns) and the faces in the algorithm's own winding
/// (the container reversed skimage's reversal), see <see cref="Run"/>.
/// </summary>
public static class MarchingCubes
{
    private const double FltEpsilon = 2.220446049250313e-16; // np.spacing(1.0)

    private static readonly Lut EdgesRelX = new(new sbyte[,] { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });
    private static readonly Lut EdgesRelY = new(new sbyte[,] { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 0 }, { 1, 1 }, { 1, 1 } });
    private static readonly Lut EdgesRelZ = new(new sbyte[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } });

    /// <summary>
    /// The surface where <paramref name="volume"/> (n0 x n1 x n2, row-major, the last axis the fastest) crosses <paramref name="isovalue"/>. Vertices are
    /// <c>(i0, i1, i2)</c> in voxel units, with the volume's values above the level inside (as <c>gradient_direction="descent"</c> has it); faces are the
    /// triangles as the cube tables give them.
    /// </summary>
    /// <exception cref="NoSurfaceException">The level is outside the range of the volume or no surface is found.</exception>
    public static SurfaceMesh Run(float[] volume, int n0, int n1, int n2, double isovalue)
    {
        if (n0 < 2 || n1 < 2 || n2 < 2) throw new ArgumentException("the volume must be at least 2x2x2");
        if (volume.Length != (long)n0 * n1 * n2) throw new ArgumentException("the volume does not have n0 * n1 * n2 values");
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in volume)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        if (isovalue < min || isovalue > max) throw new NoSurfaceException("Surface level must be within volume data range.");

        var cell = new Cell(n2, n1, n0); // Nx = last axis
        int nx = n2, ny = n1, nz = n0;
        const int st = 1;
        int nxBound = nx - 2 * st, nyBound = ny - 2 * st, nzBound = nz - 2 * st;
        double At(int z, int y, int x) => volume[(z * ny + y) * nx + x];

        var z = -st;
        while (z < nzBound)
        {
            z += st;
            var zSt = z + st;
            cell.NewZValue();
            var y = -st;
            while (y < nyBound)
            {
                y += st;
                var ySt = y + st;
                var x = -st;
                while (x < nxBound)
                {
                    x += st;
                    var xSt = x + st;
                    cell.SetCube(isovalue, x, y, z, st,
                        At(z, y, x), At(z, y, xSt), At(z, ySt, xSt), At(z, ySt, x),
                        At(zSt, y, x), At(zSt, y, xSt), At(zSt, ySt, xSt), At(zSt, ySt, x));
                    var c = LewinerTables.CASES.Get2(cell.Index, 0);
                    if (c > 0)
                    {
                        var config = LewinerTables.CASES.Get2(cell.Index, 1);
                        TheBigSwitch(cell, c, config);
                    }
                }
            }
        }

        if (cell.VertexCount == 0) throw new NoSurfaceException("No surface found at the given iso value.");
        // the cython gives (x, y, z) with x along the last axis; skimage's wrapper flips them to (axis 0, axis 1, axis 2) and the reference flips twice more
        // (the stand-in and the helper), ending in that same order; its faces are skimage's flipped ones flipped back, i.e. the cython's own
        var native = cell.Vertices;
        var flipped = new float[cell.VertexCount * 3];
        for (var i = 0; i < cell.VertexCount; i++)
        {
            flipped[i * 3] = native[i * 3 + 2];
            flipped[i * 3 + 1] = native[i * 3 + 1];
            flipped[i * 3 + 2] = native[i * 3];
        }
        return new SurfaceMesh(flipped, cell.Faces);
    }

    private static void TheBigSwitch(Cell cell, int c, int config)
    {
        var subconfig = 0;
        switch (c)
        {
            case 1: cell.AddTriangles(LewinerTables.TILING1, config, 1); break;
            case 2: cell.AddTriangles(LewinerTables.TILING2, config, 2); break;
            case 3:
                if (TestFace(cell, LewinerTables.TEST3.Get1(config))) cell.AddTriangles(LewinerTables.TILING3_2, config, 4);
                else cell.AddTriangles(LewinerTables.TILING3_1, config, 2);
                break;
            case 4:
                if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST4.Get1(config))) cell.AddTriangles(LewinerTables.TILING4_1, config, 2);
                else cell.AddTriangles(LewinerTables.TILING4_2, config, 6);
                break;
            case 5: cell.AddTriangles(LewinerTables.TILING5, config, 3); break;
            case 6:
                if (TestFace(cell, LewinerTables.TEST6.Get2(config, 0))) cell.AddTriangles(LewinerTables.TILING6_2, config, 5);
                else if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST6.Get2(config, 1))) cell.AddTriangles(LewinerTables.TILING6_1_1, config, 3);
                else cell.AddTriangles(LewinerTables.TILING6_1_2, config, 9);
                break;
            case 7:
                if (TestFace(cell, LewinerTables.TEST7.Get2(config, 0))) subconfig += 1;
                if (TestFace(cell, LewinerTables.TEST7.Get2(config, 1))) subconfig += 2;
                if (TestFace(cell, LewinerTables.TEST7.Get2(config, 2))) subconfig += 4;
                switch (subconfig)
                {
                    case 0: cell.AddTriangles(LewinerTables.TILING7_1, config, 3); break;
                    case 1: cell.AddTriangles2(LewinerTables.TILING7_2, config, 0, 5); break;
                    case 2: cell.AddTriangles2(LewinerTables.TILING7_2, config, 1, 5); break;
                    case 3: cell.AddTriangles2(LewinerTables.TILING7_3, config, 0, 9); break;
                    case 4: cell.AddTriangles2(LewinerTables.TILING7_2, config, 2, 5); break;
                    case 5: cell.AddTriangles2(LewinerTables.TILING7_3, config, 1, 9); break;
                    case 6: cell.AddTriangles2(LewinerTables.TILING7_3, config, 2, 9); break;
                    case 7:
                        if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST7.Get2(config, 3))) cell.AddTriangles(LewinerTables.TILING7_4_2, config, 9);
                        else cell.AddTriangles(LewinerTables.TILING7_4_1, config, 5);
                        break;
                }
                break;
            case 8: cell.AddTriangles(LewinerTables.TILING8, config, 2); break;
            case 9: cell.AddTriangles(LewinerTables.TILING9, config, 4); break;
            case 10:
                if (TestFace(cell, LewinerTables.TEST10.Get2(config, 0)))
                {
                    if (TestFace(cell, LewinerTables.TEST10.Get2(config, 1))) cell.AddTriangles(LewinerTables.TILING10_1_1_, config, 4);
                    else cell.AddTriangles(LewinerTables.TILING10_2, config, 8);
                }
                else if (TestFace(cell, LewinerTables.TEST10.Get2(config, 1))) cell.AddTriangles(LewinerTables.TILING10_2_, config, 8);
                else if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST10.Get2(config, 2))) cell.AddTriangles(LewinerTables.TILING10_1_1, config, 4);
                else cell.AddTriangles(LewinerTables.TILING10_1_2, config, 8);
                break;
            case 11: cell.AddTriangles(LewinerTables.TILING11, config, 4); break;
            case 12:
                if (TestFace(cell, LewinerTables.TEST12.Get2(config, 0)))
                {
                    if (TestFace(cell, LewinerTables.TEST12.Get2(config, 1))) cell.AddTriangles(LewinerTables.TILING12_1_1_, config, 4);
                    else cell.AddTriangles(LewinerTables.TILING12_2, config, 8);
                }
                else if (TestFace(cell, LewinerTables.TEST12.Get2(config, 1))) cell.AddTriangles(LewinerTables.TILING12_2_, config, 8);
                else if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST12.Get2(config, 2))) cell.AddTriangles(LewinerTables.TILING12_1_1, config, 4);
                else cell.AddTriangles(LewinerTables.TILING12_1_2, config, 8);
                break;
            case 13:
                for (var i = 0; i < 6; i++)
                    if (TestFace(cell, LewinerTables.TEST13.Get2(config, i))) subconfig += 1 << i;
                subconfig = LewinerTables.SUBCONFIG13.Get1(subconfig);
                if (subconfig == 0) cell.AddTriangles(LewinerTables.TILING13_1, config, 4);
                else if (subconfig <= 6) cell.AddTriangles2(LewinerTables.TILING13_2, config, subconfig - 1, 6);
                else if (subconfig <= 18) cell.AddTriangles2(LewinerTables.TILING13_3, config, subconfig - 7, 10);
                else if (subconfig <= 22) cell.AddTriangles2(LewinerTables.TILING13_4, config, subconfig - 19, 12);
                else if (subconfig <= 26)
                {
                    subconfig -= 23; // the original sets the subconfig "without apparent reason": it is the index of the variant
                    if (TestInternal(cell, c, config, subconfig, LewinerTables.TEST13.Get2(config, 6))) cell.AddTriangles2(LewinerTables.TILING13_5_1, config, subconfig, 6);
                    else cell.AddTriangles2(LewinerTables.TILING13_5_2, config, subconfig, 10);
                }
                else if (subconfig <= 38) cell.AddTriangles2(LewinerTables.TILING13_3_, config, subconfig - 27, 10);
                else if (subconfig <= 44) cell.AddTriangles2(LewinerTables.TILING13_2_, config, subconfig - 39, 6);
                else if (subconfig == 45) cell.AddTriangles(LewinerTables.TILING13_1_, config, 4);
                else throw new InvalidOperationException("Marching Cubes: impossible case 13");
                break;
            case 14: cell.AddTriangles(LewinerTables.TILING14, config, 4); break;
        }
    }

    /// <summary>True if the face contains part of the surface.</summary>
    private static bool TestFace(Cell cell, int face)
    {
        var absFace = face < 0 ? -face : face;
        double a = 0, b = 0, c = 0, d = 0;
        switch (absFace)
        {
            case 1: (a, b, c, d) = (cell.V[0], cell.V[4], cell.V[5], cell.V[1]); break;
            case 2: (a, b, c, d) = (cell.V[1], cell.V[5], cell.V[6], cell.V[2]); break;
            case 3: (a, b, c, d) = (cell.V[2], cell.V[6], cell.V[7], cell.V[3]); break;
            case 4: (a, b, c, d) = (cell.V[3], cell.V[7], cell.V[4], cell.V[0]); break;
            case 5: (a, b, c, d) = (cell.V[0], cell.V[3], cell.V[2], cell.V[1]); break;
            case 6: (a, b, c, d) = (cell.V[4], cell.V[7], cell.V[6], cell.V[5]); break;
        }
        var acBd = a * c - b * d;
        if (acBd > -FltEpsilon && acBd < FltEpsilon) return face >= 0;
        return face * a * acBd >= 0; // face and A invert signs
    }

    /// <summary>The interior test of the ambiguous cases 4, 6, 7, 10, 12 and 13.</summary>
    private static bool TestInternal(Cell cell, int c, int config, int subconfig, int s)
    {
        var v = cell.V;
        double t, at = 0, bt = 0, ct = 0, dt = 0;
        var test = 0;

        if (c == 4 || c == 10)
        {
            var a = (v[4] - v[0]) * (v[6] - v[2]) - (v[7] - v[3]) * (v[5] - v[1]);
            var b = v[2] * (v[4] - v[0]) + v[0] * (v[6] - v[2]) - v[1] * (v[7] - v[3]) - v[3] * (v[5] - v[1]);
            t = -b / (2 * a + FltEpsilon);
            if (t < 0 || t > 1) return s > 0;
            at = v[0] + (v[4] - v[0]) * t;
            bt = v[3] + (v[7] - v[3]) * t;
            ct = v[2] + (v[6] - v[2]) * t;
            dt = v[1] + (v[5] - v[1]) * t;
        }
        else if (c == 6 || c == 7 || c == 12 || c == 13)
        {
            var edge = c switch
            {
                6 => LewinerTables.TEST6.Get2(config, 2),
                7 => LewinerTables.TEST7.Get2(config, 4),
                12 => LewinerTables.TEST12.Get2(config, 3),
                _ => LewinerTables.TILING13_5_1.Get3(config, subconfig, 0),
            };
            switch (edge)
            {
                case 0: t = v[0] / (v[0] - v[1] + FltEpsilon); at = 0; bt = v[3] + (v[2] - v[3]) * t; ct = v[7] + (v[6] - v[7]) * t; dt = v[4] + (v[5] - v[4]) * t; break;
                case 1: t = v[1] / (v[1] - v[2] + FltEpsilon); at = 0; bt = v[0] + (v[3] - v[0]) * t; ct = v[4] + (v[7] - v[4]) * t; dt = v[5] + (v[6] - v[5]) * t; break;
                case 2: t = v[2] / (v[2] - v[3] + FltEpsilon); at = 0; bt = v[1] + (v[0] - v[1]) * t; ct = v[5] + (v[4] - v[5]) * t; dt = v[6] + (v[7] - v[6]) * t; break;
                case 3: t = v[3] / (v[3] - v[0] + FltEpsilon); at = 0; bt = v[2] + (v[1] - v[2]) * t; ct = v[6] + (v[5] - v[6]) * t; dt = v[7] + (v[4] - v[7]) * t; break;
                case 4: t = v[4] / (v[4] - v[5] + FltEpsilon); at = 0; bt = v[7] + (v[6] - v[7]) * t; ct = v[3] + (v[2] - v[3]) * t; dt = v[0] + (v[1] - v[0]) * t; break;
                case 5: t = v[5] / (v[5] - v[6] + FltEpsilon); at = 0; bt = v[4] + (v[7] - v[4]) * t; ct = v[0] + (v[3] - v[0]) * t; dt = v[1] + (v[2] - v[1]) * t; break;
                case 6: t = v[6] / (v[6] - v[7] + FltEpsilon); at = 0; bt = v[5] + (v[4] - v[5]) * t; ct = v[1] + (v[0] - v[1]) * t; dt = v[2] + (v[3] - v[2]) * t; break;
                case 7: t = v[7] / (v[7] - v[4] + FltEpsilon); at = 0; bt = v[6] + (v[5] - v[6]) * t; ct = v[2] + (v[1] - v[2]) * t; dt = v[3] + (v[0] - v[3]) * t; break;
                case 8: t = v[0] / (v[0] - v[4] + FltEpsilon); at = 0; bt = v[3] + (v[7] - v[3]) * t; ct = v[2] + (v[6] - v[2]) * t; dt = v[1] + (v[5] - v[1]) * t; break;
                case 9: t = v[1] / (v[1] - v[5] + FltEpsilon); at = 0; bt = v[0] + (v[4] - v[0]) * t; ct = v[3] + (v[7] - v[3]) * t; dt = v[2] + (v[6] - v[2]) * t; break;
                case 10: t = v[2] / (v[2] - v[6] + FltEpsilon); at = 0; bt = v[1] + (v[5] - v[1]) * t; ct = v[0] + (v[4] - v[0]) * t; dt = v[3] + (v[7] - v[3]) * t; break;
                case 11: t = v[3] / (v[3] - v[7] + FltEpsilon); at = 0; bt = v[2] + (v[6] - v[2]) * t; ct = v[1] + (v[5] - v[1]) * t; dt = v[0] + (v[4] - v[0]) * t; break;
                default: throw new InvalidOperationException($"Marching Cubes: invalid edge {edge}");
            }
        }
        else throw new InvalidOperationException($"Marching Cubes: invalid ambiguous case {c}");

        if (at >= 0) test += 1;
        if (bt >= 0) test += 2;
        if (ct >= 0) test += 4;
        if (dt >= 0) test += 8;

        switch (test)
        {
            case 0: case 1: case 2: case 3: case 4: return s > 0;
            case 5: return at * ct - bt * dt < FltEpsilon ? s > 0 : false; // the cython falls out of the function with 0 when the condition fails
            case 6: return s > 0;
            case 7: return s < 0;
            case 8: case 9: return s > 0;
            case 10: return at * ct - bt * dt >= FltEpsilon ? s > 0 : false;
            case 11: return s < 0;
            case 12: return s > 0;
            default: return s < 0; // 13, 14, 15
        }
    }

    /// <summary>The cube being visited, and the vertices and faces made so far (the <c>Cell</c> of the cython).</summary>
    private sealed class Cell
    {
        private int _x, _y, _z, _step;
        public readonly double[] V = new double[8];  // the eight corners, the isovalue subtracted
        private readonly double[] _vv = new double[8];
        private readonly double[] _vg = new double[24];
        private double _vmax;
        private double _v12X, _v12Y, _v12Z;
        private bool _v12Calculated;
        public int Index;
        private readonly int _nx, _ny;
        private int[] _faceLayer = null!;
        private int[] _faceLayer1, _faceLayer2;
        private readonly List<float> _vertices = new(1 << 16);
        private readonly List<int> _faces = new(1 << 16);

        public Cell(int nx, int ny, int nz)
        {
            _nx = nx;
            _ny = ny;
            _faceLayer1 = new int[nx * ny * 4];
            _faceLayer2 = new int[nx * ny * 4];
            Array.Fill(_faceLayer1, -1);
            Array.Fill(_faceLayer2, -1);
            _faceLayer = _faceLayer1;
        }

        public int VertexCount => _vertices.Count / 3;
        public float[] Vertices => _vertices.ToArray();
        public int[] Faces => _faces.ToArray();

        public void NewZValue()
        {
            (_faceLayer1, _faceLayer2) = (_faceLayer2, _faceLayer1);
            Array.Fill(_faceLayer2, -1);
        }

        public void SetCube(double isovalue, int x, int y, int z, int step, double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7)
        {
            _x = x; _y = y; _z = z; _step = step;
            V[0] = v0 - isovalue; V[1] = v1 - isovalue; V[2] = v2 - isovalue; V[3] = v3 - isovalue;
            V[4] = v4 - isovalue; V[5] = v5 - isovalue; V[6] = v6 - isovalue; V[7] = v7 - isovalue;
            var index = 0;
            if (V[0] > 0.0) index += 1;
            if (V[1] > 0.0) index += 2;
            if (V[2] > 0.0) index += 4;
            if (V[3] > 0.0) index += 8;
            if (V[4] > 0.0) index += 16;
            if (V[5] > 0.0) index += 32;
            if (V[6] > 0.0) index += 64;
            if (V[7] > 0.0) index += 128;
            Index = index;
            _v12Calculated = false;
        }

        public void AddTriangles(Lut lut, int lutIndex, int nt)
        {
            PrepareForAddingTriangles();
            for (var i = 0; i < nt; i++)
                for (var j = 0; j < 3; j++) AddFaceFromEdgeIndex(lut.Get2(lutIndex, i * 3 + j));
        }

        public void AddTriangles2(Lut lut, int lutIndex, int lutIndex2, int nt)
        {
            PrepareForAddingTriangles();
            for (var i = 0; i < nt; i++)
                for (var j = 0; j < 3; j++) AddFaceFromEdgeIndex(lut.Get3(lutIndex, lutIndex2, i * 3 + j));
        }

        private int AddVertex(float x, float y, float z)
        {
            _vertices.Add(x);
            _vertices.Add(y);
            _vertices.Add(z);
            return _vertices.Count / 3 - 1;
        }

        private void AddFaceFromEdgeIndex(int vi)
        {
            var indexInFaceLayer = GetIndexInFaceLayer(vi);
            var indexInVertexArray = _faceLayer[indexInFaceLayer];

            if (vi == 12) // the vertex of the centre
            {
                if (!_v12Calculated) CalculateCenterVertex();
                if (indexInVertexArray >= 0)
                {
                    _faces.Add(indexInVertexArray);
                }
                else
                {
                    indexInVertexArray = AddVertex((float)_v12X, (float)_v12Y, (float)_v12Z);
                    _faceLayer[indexInFaceLayer] = indexInVertexArray;
                    _faces.Add(indexInVertexArray);
                }
                return;
            }

            if (indexInVertexArray >= 0)
            {
                _faces.Add(indexInVertexArray); // the vertex exists: only the face
                return;
            }

            int dx1 = EdgesRelX.Get2(vi, 0), dx2 = EdgesRelX.Get2(vi, 1);
            int dy1 = EdgesRelY.Get2(vi, 0), dy2 = EdgesRelY.Get2(vi, 1);
            int dz1 = EdgesRelZ.Get2(vi, 0), dz2 = EdgesRelZ.Get2(vi, 1);
            var index1 = dz1 * 4 + dy1 * 2 + dx1;
            var index2 = dz2 * 4 + dy2 * 2 + dx2;
            var tmpf1 = 1.0 / (FltEpsilon + Math.Abs(_vv[index1]));
            var tmpf2 = 1.0 / (FltEpsilon + Math.Abs(_vv[index2]));
            double stp = _step;

            // interpolate by a kind of centre of mass
            double fx = 0.0, fy = 0.0, fz = 0.0, ff = 0.0;
            fx += dx1 * tmpf1; fy += dy1 * tmpf1; fz += dz1 * tmpf1; ff += tmpf1;
            fx += dx2 * tmpf2; fy += dy2 * tmpf2; fz += dz2 * tmpf2; ff += tmpf2;

            indexInVertexArray = AddVertex((float)(_x + stp * fx / ff), (float)(_y + stp * fy / ff), (float)(_z + stp * fz / ff));
            _faceLayer[indexInFaceLayer] = indexInVertexArray;
            _faces.Add(indexInVertexArray);
        }

        /// <summary>Where a vertex lives in the face layers (4 per cell: the vertices of the 3 edges that leave its corner, and the centre); sets <see cref="_faceLayer"/>.</summary>
        private int GetIndexInFaceLayer(int vi)
        {
            var i = _nx * _y + _x;
            var j = 0;
            int[] faceLayer;
            if (vi < 8)
            {
                if (vi < 4) faceLayer = _faceLayer1;
                else
                {
                    vi -= 4;
                    faceLayer = _faceLayer2;
                }
                if (vi == 1) { i += _step; j = 1; }
                else if (vi == 2) i += _nx * _step;
                else if (vi == 3) j = 1;
            }
            else if (vi < 12)
            {
                faceLayer = _faceLayer1;
                j = 2;
                if (vi == 9) i += _step;
                else if (vi == 10) i += _nx * _step + _step;
                else if (vi == 11) i += _nx * _step;
            }
            else
            {
                faceLayer = _faceLayer1;
                j = 3;
            }
            _faceLayer = faceLayer;
            return 4 * i + j;
        }

        private void PrepareForAddingTriangles()
        {
            // the corner values in an indexable array (the numbering is not the bitwise OR of x y z)
            _vv[0] = V[0]; _vv[1] = V[1]; _vv[2] = V[3]; _vv[3] = V[2];
            _vv[4] = V[4]; _vv[5] = V[5]; _vv[6] = V[7]; _vv[7] = V[6];
            double vmin = 0.0, vmax = 0.0;
            for (var i = 0; i < 8; i++)
            {
                if (_vv[i] > vmax) vmax = _vv[i];
                if (_vv[i] < vmin) vmin = _vv[i];
            }
            _vmax = vmax - vmin;

            // gradients at the corners (only the centre vertex uses them here)
            _vg[0 * 3 + 0] = V[0] - V[1]; _vg[0 * 3 + 1] = V[0] - V[3]; _vg[0 * 3 + 2] = V[0] - V[4];
            _vg[1 * 3 + 0] = V[0] - V[1]; _vg[1 * 3 + 1] = V[1] - V[2]; _vg[1 * 3 + 2] = V[1] - V[5];
            _vg[2 * 3 + 0] = V[3] - V[2]; _vg[2 * 3 + 1] = V[1] - V[2]; _vg[2 * 3 + 2] = V[2] - V[6];
            _vg[3 * 3 + 0] = V[3] - V[2]; _vg[3 * 3 + 1] = V[0] - V[3]; _vg[3 * 3 + 2] = V[3] - V[7];
            _vg[4 * 3 + 0] = V[4] - V[5]; _vg[4 * 3 + 1] = V[4] - V[7]; _vg[4 * 3 + 2] = V[0] - V[4];
            _vg[5 * 3 + 0] = V[4] - V[5]; _vg[5 * 3 + 1] = V[5] - V[6]; _vg[5 * 3 + 2] = V[1] - V[5];
            _vg[6 * 3 + 0] = V[7] - V[6]; _vg[6 * 3 + 1] = V[5] - V[6]; _vg[6 * 3 + 2] = V[2] - V[6];
            _vg[7 * 3 + 0] = V[7] - V[6]; _vg[7 * 3 + 1] = V[4] - V[7]; _vg[7 * 3 + 2] = V[3] - V[7];
            _ = _vmax;
        }

        private void CalculateCenterVertex()
        {
            double fx = 0.0, fy = 0.0, fz = 0.0, ff = 0.0;
            var s0 = 1.0 / (FltEpsilon + Math.Abs(V[0]));
            var s1 = 1.0 / (FltEpsilon + Math.Abs(V[1]));
            var s2 = 1.0 / (FltEpsilon + Math.Abs(V[2]));
            var s3 = 1.0 / (FltEpsilon + Math.Abs(V[3]));
            var s4 = 1.0 / (FltEpsilon + Math.Abs(V[4]));
            var s5 = 1.0 / (FltEpsilon + Math.Abs(V[5]));
            var s6 = 1.0 / (FltEpsilon + Math.Abs(V[6]));
            var s7 = 1.0 / (FltEpsilon + Math.Abs(V[7]));

            fx += 0.0 * s0; fy += 0.0 * s0; fz += 0.0 * s0; ff += s0;
            fx += 1.0 * s1; fy += 0.0 * s1; fz += 0.0 * s1; ff += s1;
            fx += 1.0 * s2; fy += 1.0 * s2; fz += 0.0 * s2; ff += s2;
            fx += 0.0 * s3; fy += 1.0 * s3; fz += 0.0 * s3; ff += s3;
            fx += 0.0 * s4; fy += 0.0 * s4; fz += 1.0 * s4; ff += s4;
            fx += 1.0 * s5; fy += 0.0 * s5; fz += 1.0 * s5; ff += s5;
            fx += 1.0 * s6; fy += 1.0 * s6; fz += 1.0 * s6; ff += s6;
            fx += 0.0 * s7; fy += 1.0 * s7; fz += 1.0 * s7; ff += s7;

            double stp = _step;
            _v12X = _x + stp * fx / ff;
            _v12Y = _y + stp * fy / ff;
            _v12Z = _z + stp * fz / ff;
            _v12Calculated = true;
        }
    }
}
