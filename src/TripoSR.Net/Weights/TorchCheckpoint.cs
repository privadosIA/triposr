using System.Text;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

// Copied from ~/Proyectos/lion-clap (LionClap.Weights): to be moved to musicgen-core and shared by both (see docs/porting-plan.md).
namespace TripoSR.Weights;

/// <summary>
/// A PyTorch checkpoint (<c>pytorch_model.bin</c>: a zip of stored entries holding a pickled state_dict and one raw storage per tensor), read without
/// Python and without running anything from the file. Opening reads only the zip directory and the pickle (the headers); the tensors are read on demand
/// with <see cref="SafeTensors.ReadTensor"/> from their absolute position, so their bytes are exactly the file's. Float32 and int64 tensors in row-major
/// layout are supported.
/// </summary>
public sealed class TorchCheckpoint
{
    private const int ElementSize32 = 4, ElementSize64 = 8;

    /// <summary>Every tensor of the state_dict: name, dtype (<c>F32</c>/<c>I64</c>), shape and absolute file position.</summary>
    public IReadOnlyDictionary<string, TensorInfo> Tensors { get; }

    public string Path { get; }

    private TorchCheckpoint(string path, IReadOnlyDictionary<string, TensorInfo> tensors) => (Path, Tensors) = (path, tensors);

    public static TorchCheckpoint Open(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint was not found at {path}", path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return new TorchCheckpoint(path, Parse(stream, path));
    }

    /// <summary>Opens from a stream (for tests that count reads); <paramref name="file"/> is the name recorded in the tensor positions.</summary>
    internal static TorchCheckpoint Open(Stream stream, string file) => new(file, Parse(stream, file));

    private static Dictionary<string, TensorInfo> Parse(Stream stream, string file)
    {
        var entries = ZipDirectory.Read(stream);
        var pickles = entries.Where(e => e.Name.EndsWith("/data.pkl", StringComparison.Ordinal) || e.Name == "data.pkl").ToList();
        if (pickles.Count != 1)
            throw new InvalidDataException(pickles.Count == 0
                ? "not a readable PyTorch checkpoint: the zip has no data.pkl"
                : "not a readable PyTorch checkpoint: the zip has more than one data.pkl");
        var prefix = pickles[0].Name[..^"data.pkl".Length]; // "pytorch_model/"

        var order = entries.FirstOrDefault(e => e.Name == prefix + "byteorder");
        if (order is not null && Encoding.ASCII.GetString(ZipDirectory.ReadAt(stream, order.DataOffset, (int)Math.Min(order.Size, 16))).Trim() != "little")
            throw new NotSupportedException("only little-endian PyTorch checkpoints are supported");

        var storages = entries.Where(e => e.Name.StartsWith(prefix + "data/", StringComparison.Ordinal))
            .ToDictionary(e => e.Name[(prefix.Length + "data/".Length)..], e => e);
        if (pickles[0].Size > 256 * 1024 * 1024) throw new InvalidDataException("the pickle is implausibly large");
        var pickle = ZipDirectory.ReadAt(stream, pickles[0].DataOffset, (int)pickles[0].Size);

        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);
        foreach (var (name, t) in PickleReader.ReadStateDict(pickle))
        {
            if (tensors.ContainsKey(name)) throw new InvalidDataException($"the state_dict has {name} twice");
            if (!storages.TryGetValue(t.StorageKey, out var storage))
                throw new InvalidDataException($"{name}: its storage data/{t.StorageKey} is not in the zip");
            var element = t.Dtype == "F32" ? ElementSize32 : ElementSize64;
            if (storage.Size != t.StorageElements * element)
                throw new InvalidDataException($"{name}: storage data/{t.StorageKey} holds {storage.Size} bytes, the pickle says {t.StorageElements} elements");

            if (!IsRowMajor(t.Shape, t.Stride))
                throw new NotSupportedException($"{name}: the tensor is not contiguous (strides [{string.Join(", ", t.Stride)}] for shape [{string.Join(", ", t.Shape)}])");
            var count = t.Shape.Aggregate(1L, (a, b) => checked(a * b));
            var end = checked((t.Offset + count) * element);
            if (end > storage.Size)
                throw new InvalidDataException($"{name}: the tensor points past the end of its storage data/{t.StorageKey}");
            tensors[name] = new TensorInfo(name, file, t.Dtype, t.Shape, storage.DataOffset + t.Offset * element, count * element);
        }
        return tensors;
    }

    // dimensions of size 1 may have any stride
    private static bool IsRowMajor(long[] shape, long[] stride)
    {
        long expected = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] != 1 && stride[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    /// <summary>Reads <paramref name="key"/> as stored (CPU, the file's dtype).</summary>
    public Tensor Read(string key) =>
        Tensors.TryGetValue(key, out var info)
            ? SafeTensors.ReadTensor(info)
            : throw new KeyNotFoundException($"The checkpoint does not contain the tensor {key}");

    /// <summary>Reads <paramref name="key"/> converted to <paramref name="dtype"/>.</summary>
    public Tensor Read(string key, ScalarType dtype)
    {
        var raw = Read(key);
        if (raw.dtype == dtype) return raw;
        try { return raw.to(dtype); }
        finally { raw.Dispose(); }
    }
}
