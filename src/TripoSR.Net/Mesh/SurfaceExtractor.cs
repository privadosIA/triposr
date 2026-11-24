using TorchSharp;
using TripoSR.Field;
using static TorchSharp.torch;

namespace TripoSR.Mesh;

/// <summary>A triangle mesh with a colour per vertex (RGBA, 8 bits).</summary>
public sealed record ColoredMesh(float[] Vertices, int[] Faces, byte[] Colors)
{
    public int VertexCount => Vertices.Length / 3;
    public int TriangleCount => Faces.Length / 3;
}

/// <summary>
/// <c>TSR.extract_mesh</c>: the density of the field on a grid of <c>resolution</c> points a side across the cube of side 2 * radius, marching cubes at the threshold
/// (the reference's <c>MarchingCubeHelper</c>: the surface where the density crosses 25, in its own order of axes and winding), and the colour of the field at each
/// vertex.
/// </summary>
public static class SurfaceExtractor
{
    public const float Threshold = 25.0f;

    /// <summary>Marching cubes on the density grid. Returns the vertices in the cube (-radius, radius) and the faces.</summary>
    /// <exception cref="InvalidOperationException">The surface is empty: no density crosses the threshold.</exception>
    public static SurfaceMesh Surface(NeRFMLP decoder, Tensor sceneCode, int resolution, TripoSRConfig? config = null, float threshold = Threshold)
    {
        var c = config ?? new TripoSRConfig();
        using var _ = no_grad();
        using var density = Density(decoder, sceneCode, resolution, c);
        return Surface(density, resolution, c, threshold);
    }

    /// <summary>The density (after bias and exp) on the grid, flat (R*R*R, 1), index <c>ix * R * R + iy * R + iz</c>.</summary>
    public static Tensor Density(NeRFMLP decoder, Tensor sceneCode, int resolution, TripoSRConfig? config = null)
    {
        var c = config ?? new TripoSRConfig();
        using var scope = NewDisposeScope();
        var grid = GridVertices(resolution).to(sceneCode.device);                    // in 0..1
        var scaled = TriplaneField.ScaleTensor(grid, 0, 1, -c.Radius, c.Radius);    // in (-radius, radius)
        return TriplaneField.QueryDensity(decoder, scaled, sceneCode, c).MoveToOuterDisposeScope();
    }

    /// <summary>Marching cubes over a density grid made by <see cref="Density"/>.</summary>
    public static SurfaceMesh Surface(Tensor density, int resolution, TripoSRConfig? config = null, float threshold = Threshold)
    {
        var c = config ?? new TripoSRConfig();
        using var _ = no_grad();
        // the reference hands marching cubes -(density - threshold) and the helper negates it again: the volume is density - threshold, the level 0
        using var level = (density.reshape(resolution, resolution, resolution) - threshold).contiguous();
        var volume = level.cpu().data<float>().ToArray();
        var mesh = MarchingCubes.Run(volume, resolution, resolution, resolution, 0.0);
        // v_pos / (resolution - 1), then from the range (0, 1) to (-radius, radius)
        using var v = tensor(mesh.Vertices, [mesh.VertexCount, 3]).to(density.device);
        using var unit = v / (resolution - 1.0);
        using var vertices = TriplaneField.ScaleTensor(unit, 0, 1, -c.Radius, c.Radius);
        return new SurfaceMesh(vertices.cpu().data<float>().ToArray(), mesh.Faces);
    }

    /// <summary>The colour of the field at the vertices of a surface, as 8-bit RGBA with alpha 255 (what trimesh makes of float colours).</summary>
    public static byte[] VertexColors(NeRFMLP decoder, Tensor sceneCode, SurfaceMesh surface, TripoSRConfig? config = null)
    {
        using var _ = no_grad();
        using var positions = tensor(surface.Vertices, [surface.VertexCount, 3]).to(sceneCode.device);
        using var sample = TriplaneField.Query(decoder, positions, sceneCode, config);
        var rgb = sample.Color.cpu().contiguous().data<float>().ToArray();
        var rgba = new byte[surface.VertexCount * 4];
        for (var i = 0; i < surface.VertexCount; i++)
        {
            for (var k = 0; k < 3; k++) rgba[i * 4 + k] = TrimeshColor(rgb[i * 3 + k]);
            rgba[i * 4 + 3] = 255;
        }
        return rgba;
    }

    /// <summary>The whole step of the reference: surface and colours, with the coincident vertices merged as trimesh does when it builds the mesh.</summary>
    public static ColoredMesh Extract(NeRFMLP decoder, Tensor sceneCode, int resolution, TripoSRConfig? config = null, float threshold = Threshold)
    {
        var surface = Surface(decoder, sceneCode, resolution, config, threshold);
        return MeshCleanup.MergeVertices(new ColoredMesh(surface.Vertices, surface.Faces, VertexColors(decoder, sceneCode, surface, config)));
    }

    // trimesh.visual.color.to_rgba: float colours in 0..1 become uint8 by (c * 255).round() (see the colour test against the reference)
    internal static byte TrimeshColor(float c) => (byte)Math.Clamp((int)MathF.Round(c * 255f, MidpointRounding.ToEven), 0, 255);

    /// <summary>The grid of the helper: <c>linspace(0, 1, R)</c> on each axis, <c>meshgrid(indexing="ij")</c>, as (R*R*R, 3).</summary>
    public static Tensor GridVertices(int resolution)
    {
        using var a = linspace(0, 1, resolution);
        var g = meshgrid([a, a, a], indexing: "ij");
        try
        {
            return stack([g[0].reshape(-1), g[1].reshape(-1), g[2].reshape(-1)], -1);
        }
        finally { foreach (var t in g) t.Dispose(); }
    }
}
