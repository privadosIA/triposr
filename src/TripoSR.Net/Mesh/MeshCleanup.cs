namespace TripoSR.Mesh;

/// <summary>What <c>trimesh.Trimesh(..., process=True)</c> does to the mesh of the reference when it is made: vertices that fall on the same place are merged.</summary>
public static class MeshCleanup
{
    /// <summary>
    /// <c>merge_vertices</c> with trimesh's default tolerance: vertices equal after rounding their coordinates to 1e-8 become one (the first keeps its colour), the faces are
    /// renumbered and none is removed. The vertices keep their order of first appearance (trimesh leaves it to a hash; nothing depends on it).
    /// </summary>
    public static ColoredMesh MergeVertices(ColoredMesh mesh)
    {
        const double Scale = 1e8; // 10 ** digits, digits = 8 for tol.merge = 1e-8
        var n = mesh.VertexCount;
        var map = new Dictionary<(long, long, long), int>(n);
        var remap = new int[n];
        var keep = new List<int>(n);
        for (var i = 0; i < n; i++)
        {
            var key = ((long)Math.Round(mesh.Vertices[i * 3] * Scale), (long)Math.Round(mesh.Vertices[i * 3 + 1] * Scale), (long)Math.Round(mesh.Vertices[i * 3 + 2] * Scale));
            if (map.TryGetValue(key, out var existing)) remap[i] = existing;
            else
            {
                remap[i] = keep.Count;
                map[key] = keep.Count;
                keep.Add(i);
            }
        }
        if (keep.Count == n) return mesh;
        var vertices = new float[keep.Count * 3];
        var colors = new byte[keep.Count * 4];
        for (var k = 0; k < keep.Count; k++)
        {
            Array.Copy(mesh.Vertices, keep[k] * 3, vertices, k * 3, 3);
            Array.Copy(mesh.Colors, keep[k] * 4, colors, k * 4, 4);
        }
        var faces = new int[mesh.Faces.Length];
        for (var i = 0; i < faces.Length; i++) faces[i] = remap[mesh.Faces[i]];
        return new ColoredMesh(vertices, faces, colors);
    }
}
