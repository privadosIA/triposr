namespace TripoSR.Mesh;

/// <summary>A look-up table of the marching cubes: signed bytes with up to three indices, row-major (the <c>Lut</c> of scikit-image's cython).</summary>
internal sealed class Lut
{
    private readonly sbyte[] _values;
    private readonly int _l1, _l2;

    public Lut(int[] shape, string base64)
    {
        _l1 = shape.Length > 1 ? shape[1] : 1;
        _l2 = shape.Length > 2 ? shape[2] : 1;
        var bytes = Convert.FromBase64String(base64);
        _values = new sbyte[bytes.Length];
        Buffer.BlockCopy(bytes, 0, _values, 0, bytes.Length);
        var expected = shape[0] * _l1 * _l2;
        if (_values.Length != expected) throw new InvalidDataException($"a table of shape [{string.Join(",", shape)}] holds {_values.Length} values, {expected} expected");
    }

    public Lut(sbyte[,] values)
    {
        _l1 = values.GetLength(1);
        _l2 = 1;
        _values = new sbyte[values.Length];
        for (var i = 0; i < values.GetLength(0); i++)
            for (var j = 0; j < _l1; j++) _values[i * _l1 + j] = values[i, j];
    }

    public int Get1(int i0) => _values[i0];

    public int Get2(int i0, int i1) => _values[i0 * _l1 + i1];

    public int Get3(int i0, int i1, int i2) => _values[i0 * _l1 * _l2 + i1 * _l2 + i2];
}
