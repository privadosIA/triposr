using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace TripoSR.Mesh;

/// <summary>
/// A binary glTF 2.0 (<c>.glb</c>) of one triangle mesh with a colour per vertex, laid out as trimesh writes it (the reference's <c>mesh.export(file_type="glb")</c>): one scene,
/// one node, one mesh with one primitive (triangles; <c>POSITION</c> float32, <c>COLOR_0</c> unsigned bytes normalised, indices uint32), all in one binary buffer.
/// </summary>
public static class GlbWriter
{
    private const uint Magic = 0x46546C67, JsonChunk = 0x4E4F534A, BinChunk = 0x004E4942;

    public static byte[] Write(ColoredMesh mesh)
    {
        int nv = mesh.VertexCount, nf = mesh.Faces.Length;
        if (nv == 0 || nf == 0) throw new ArgumentException("a model needs vertices and triangles");
        if (mesh.Colors.Length != nv * 4) throw new ArgumentException("one RGBA colour per vertex is needed");

        // the binary buffer: indices, positions, colours (each a multiple of 4 bytes)
        int indexBytes = nf * 4, positionBytes = nv * 12, colorBytes = nv * 4;
        var bin = new byte[indexBytes + positionBytes + colorBytes];
        for (var i = 0; i < nf; i++) BinaryPrimitives.WriteUInt32LittleEndian(bin.AsSpan(i * 4), (uint)mesh.Faces[i]);
        for (var i = 0; i < nv * 3; i++) BinaryPrimitives.WriteSingleLittleEndian(bin.AsSpan(indexBytes + i * 4), mesh.Vertices[i]);
        Buffer.BlockCopy(mesh.Colors, 0, bin, indexBytes + positionBytes, colorBytes);

        float[] pmin = [float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity], pmax = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        for (var i = 0; i < nv * 3; i++)
        {
            var k = i % 3;
            if (mesh.Vertices[i] < pmin[k]) pmin[k] = mesh.Vertices[i];
            if (mesh.Vertices[i] > pmax[k]) pmax[k] = mesh.Vertices[i];
        }
        int[] cmin = [255, 255, 255, 255], cmax = [0, 0, 0, 0];
        for (var i = 0; i < nv * 4; i++)
        {
            var k = i % 4;
            if (mesh.Colors[i] < cmin[k]) cmin[k] = mesh.Colors[i];
            if (mesh.Colors[i] > cmax[k]) cmax[k] = mesh.Colors[i];
        }

        using var js = new MemoryStream();
        using (var w = new Utf8JsonWriter(js))
        {
            w.WriteStartObject();
            w.WriteNumber("scene", 0);
            w.WriteStartArray("scenes"); w.WriteStartObject(); w.WriteStartArray("nodes"); w.WriteNumberValue(0); w.WriteEndArray(); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartObject("asset"); w.WriteString("version", "2.0"); w.WriteString("generator", "TripoSR.Net"); w.WriteEndObject();
            w.WriteStartArray("accessors");
            w.WriteStartObject(); w.WriteNumber("componentType", 5125); w.WriteString("type", "SCALAR"); w.WriteNumber("bufferView", 0); w.WriteNumber("count", nf);
            w.WriteStartArray("max"); w.WriteNumberValue(nv - 1); w.WriteEndArray(); w.WriteStartArray("min"); w.WriteNumberValue(0); w.WriteEndArray(); w.WriteEndObject();
            w.WriteStartObject(); w.WriteNumber("componentType", 5126); w.WriteString("type", "VEC3"); w.WriteNumber("byteOffset", 0); w.WriteNumber("bufferView", 1); w.WriteNumber("count", nv);
            w.WriteStartArray("max"); foreach (var f in pmax) w.WriteNumberValue(f); w.WriteEndArray(); w.WriteStartArray("min"); foreach (var f in pmin) w.WriteNumberValue(f); w.WriteEndArray(); w.WriteEndObject();
            w.WriteStartObject(); w.WriteNumber("componentType", 5121); w.WriteBoolean("normalized", true); w.WriteString("type", "VEC4"); w.WriteNumber("byteOffset", 0); w.WriteNumber("bufferView", 2); w.WriteNumber("count", nv);
            w.WriteStartArray("max"); foreach (var c in cmax) w.WriteNumberValue(c); w.WriteEndArray(); w.WriteStartArray("min"); foreach (var c in cmin) w.WriteNumberValue(c); w.WriteEndArray(); w.WriteEndObject();
            w.WriteEndArray();
            w.WriteStartArray("meshes"); w.WriteStartObject(); w.WriteString("name", "geometry_0");
            w.WriteStartArray("primitives"); w.WriteStartObject();
            w.WriteStartObject("attributes"); w.WriteNumber("POSITION", 1); w.WriteNumber("COLOR_0", 2); w.WriteEndObject();
            w.WriteNumber("indices", 0); w.WriteNumber("mode", 4);
            w.WriteEndObject(); w.WriteEndArray(); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("nodes"); w.WriteStartObject(); w.WriteString("name", "geometry_0"); w.WriteNumber("mesh", 0); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("buffers"); w.WriteStartObject(); w.WriteNumber("byteLength", bin.Length); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("bufferViews");
            int[] offsets = [0, indexBytes, indexBytes + positionBytes], lengths = [indexBytes, positionBytes, colorBytes];
            for (var i = 0; i < 3; i++) { w.WriteStartObject(); w.WriteNumber("buffer", 0); w.WriteNumber("byteOffset", offsets[i]); w.WriteNumber("byteLength", lengths[i]); w.WriteEndObject(); }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        var json = js.ToArray();
        var jsonPadded = (json.Length + 3) / 4 * 4;
        var total = 12 + 8 + jsonPadded + 8 + bin.Length;
        var o = new byte[total];
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(0), Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(4), 2);
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(8), (uint)total);
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(12), (uint)jsonPadded);
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(16), JsonChunk);
        Buffer.BlockCopy(json, 0, o, 20, json.Length);
        for (var i = json.Length; i < jsonPadded; i++) o[20 + i] = (byte)' ';
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(20 + jsonPadded), (uint)bin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(o.AsSpan(24 + jsonPadded), BinChunk);
        Buffer.BlockCopy(bin, 0, o, 28 + jsonPadded, bin.Length);
        return o;
    }
}
