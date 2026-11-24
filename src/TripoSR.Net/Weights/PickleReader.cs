using System.Buffers.Binary;
using System.Text;

// Copied from ~/Proyectos/lion-clap (LionClap.Weights): to be moved to musicgen-core and shared by both (see docs/porting-plan.md).
namespace TripoSR.Weights;

/// <summary>A tensor as the pickle describes it: which storage, where in it, its shape and strides.</summary>
internal sealed record TensorRef(string StorageKey, string Dtype, long StorageElements, long Offset, long[] Shape, long[] Stride);

/// <summary>
/// A whitelist interpreter for the pickle of a PyTorch state_dict, not a general pickle machine. It understands the handful of opcodes that
/// <c>torch.save</c> (protocol 2) emits for a dictionary of tensors, builds only strings, integers, tuples, one ordered dictionary and
/// tensor descriptions, and refuses everything else with an error naming it. It never imports a module nor calls anything.
/// </summary>
internal sealed class PickleReader
{
    private const int MaxStack = 1 << 20, MaxMemo = 1 << 20;

    private static readonly object Mark = new();

    private sealed record GlobalRef(string Module, string Name)
    {
        public override string ToString() => Module + "." + Name;
    }

    private sealed record StorageRef(string Key, string Dtype, long Elements);

    private sealed class Dict : List<KeyValuePair<object, object>> { }

    private static readonly Dictionary<string, string> StorageDtypes = new()
    {
        ["torch.FloatStorage"] = "F32",
        ["torch.LongStorage"] = "I64",
    };

    private readonly byte[] _data;
    private int _p;
    private readonly List<object> _stack = new();
    private readonly Dictionary<int, object> _memo = new();

    private PickleReader(byte[] data) => _data = data;

    /// <summary>Interprets <paramref name="pickle"/> and returns the tensors of its top-level dictionary in file order.</summary>
    public static IReadOnlyList<KeyValuePair<string, TensorRef>> ReadStateDict(byte[] pickle) => new PickleReader(pickle).Run();

    private List<KeyValuePair<string, TensorRef>> Run()
    {
        while (true)
        {
            var at = _p;
            var op = Byte();
            switch (op)
            {
                case 0x80: Byte(); break;                                            // PROTO
                case 0x7d: Push(new Dict()); break;                                  // EMPTY_DICT
                case 0x71: Memoize(Byte()); break;                                   // BINPUT
                case 0x72: Memoize(Int32()); break;                                  // LONG_BINPUT
                case 0x68: Push(Recall(Byte())); break;                              // BINGET
                case 0x6a: Push(Recall(Int32())); break;                             // LONG_BINGET
                case 0x28: Push(Mark); break;                                        // MARK
                case 0x58: Push(Unicode()); break;                                   // BINUNICODE
                case 0x63: Push(Global()); break;                                    // GLOBAL
                case 0x4b: Push((long)Byte()); break;                                // BININT1
                case 0x4d: Push((long)UInt16()); break;                              // BININT2
                case 0x4a: Push((long)Int32()); break;                               // BININT
                case 0x8a: Push(Long1()); break;                                     // LONG1
                case 0x74: Push(TupleFromMark()); break;                             // TUPLE
                case 0x85: Push(TupleOf(1)); break;                                  // TUPLE1
                case 0x86: Push(TupleOf(2)); break;                                  // TUPLE2
                case 0x87: Push(TupleOf(3)); break;                                  // TUPLE3
                case 0x29: Push(Array.Empty<object>()); break;                       // EMPTY_TUPLE
                case 0x51: Push(Persistent(Pop())); break;                           // BINPERSID
                case 0x89: Push(false); break;                                       // NEWFALSE
                case 0x88: Push(true); break;                                        // NEWTRUE
                case 0x4e: Push(null!); break;                                       // NONE
                case 0x52: { var args = Pop(); var callable = Pop(); Push(Call(callable, args)); break; } // REDUCE
                case 0x75: SetItems(); break;                                        // SETITEMS
                case 0x73: { var v = Pop(); var k = Pop(); Top<Dict>("SETITEM").Add(new(k, v)); break; } // SETITEM
                case 0x2e: return Result(Pop());                                      // STOP
                default:
                    throw new InvalidDataException($"unsupported pickle opcode 0x{op:x2} at byte {at}: this is not a plain PyTorch state_dict");
            }
        }
    }

    private static List<KeyValuePair<string, TensorRef>> Result(object root)
    {
        if (root is not Dict dict) throw new InvalidDataException("the pickle does not hold a dictionary of tensors");
        var result = new List<KeyValuePair<string, TensorRef>>(dict.Count);
        foreach (var (k, v) in dict)
        {
            if (k is not string name) throw new InvalidDataException("the state_dict has a key that is not a string");
            if (v is not TensorRef tensor) throw new InvalidDataException($"the value of {name} is not a tensor");
            result.Add(new(name, tensor));
        }
        return result;
    }

    // ---- values --------------------------------------------------------------------------------------------------------------------------

    private static object Call(object callable, object args)
    {
        if (callable is not GlobalRef g || args is not object[] a)
            throw new InvalidDataException("the pickle calls something that is not an allowed function");
        switch (g.ToString())
        {
            case "collections.OrderedDict":
                if (a.Length != 0) throw new InvalidDataException("collections.OrderedDict is only allowed without arguments");
                return new Dict();
            case "torch._utils._rebuild_tensor_v2":
                return RebuildTensor(a);
            default:
                throw new InvalidDataException($"the pickle calls {g}, which is not allowed");
        }
    }

    private static TensorRef RebuildTensor(object[] a)
    {
        // (storage, storage_offset, size, stride, requires_grad, backward_hooks)
        if (a.Length != 6 || a[0] is not StorageRef storage || a[1] is not long offset || a[2] is not object[] size || a[3] is not object[] stride)
            throw new InvalidDataException("a tensor in the pickle does not have the expected description");
        var shape = size.Select(Dimension).ToArray();
        var strides = stride.Select(Dimension).ToArray();
        if (shape.Length != strides.Length) throw new InvalidDataException("a tensor has different numbers of sizes and strides");
        if (offset < 0) throw new InvalidDataException("a tensor has a negative storage offset");
        return new TensorRef(storage.Key, storage.Dtype, storage.Elements, offset, shape, strides);
    }

    private static long Dimension(object o) =>
        o is long n && n >= 0 ? n : throw new InvalidDataException("a tensor has a size or stride that is not a non-negative integer");

    private static StorageRef Persistent(object id)
    {
        // ('storage', <storage class>, key, location, number of elements)
        if (id is not object[] t || t.Length != 5 || t[0] is not "storage" || t[1] is not GlobalRef cls || t[2] is not string key || t[4] is not long elements)
            throw new InvalidDataException("the pickle has a persistent id that is not a storage");
        if (!StorageDtypes.TryGetValue(cls.ToString(), out var dtype))
            throw new NotSupportedException($"storage type {cls} is not supported (only float32 and int64)");
        if (elements < 0) throw new InvalidDataException("a storage has a negative size");
        return new StorageRef(key, dtype, elements);
    }

    private GlobalRef Global()
    {
        var module = Line();
        var name = Line();
        var g = new GlobalRef(module, name);
        var text = g.ToString();
        if (module == "torch" && name.EndsWith("Storage", StringComparison.Ordinal) && !StorageDtypes.ContainsKey(text))
            throw new NotSupportedException($"storage type {text} is not supported (only float32 and int64)");
        if (text is not ("collections.OrderedDict" or "torch._utils._rebuild_tensor_v2") && !StorageDtypes.ContainsKey(text))
            throw new InvalidDataException($"the pickle references {text}, which is not allowed in a PyTorch checkpoint");
        return g;
    }

    private void SetItems()
    {
        var items = PopToMark();
        if (items.Count % 2 != 0) throw new InvalidDataException("SETITEMS with an odd number of items");
        var dict = Top<Dict>("SETITEMS");
        for (var i = 0; i < items.Count; i += 2) dict.Add(new(items[i], items[i + 1]));
    }

    // ---- stack ---------------------------------------------------------------------------------------------------------------------------

    private void Push(object value)
    {
        if (_stack.Count >= MaxStack) throw new InvalidDataException("the pickle stack is too deep");
        _stack.Add(value);
    }

    private object Pop()
    {
        if (_stack.Count == 0) throw new InvalidDataException("the pickle pops an empty stack");
        var v = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        return v;
    }

    private T Top<T>(string op) where T : class =>
        _stack.Count > 0 && _stack[^1] is T t ? t : throw new InvalidDataException($"{op} needs a {typeof(T).Name.ToLowerInvariant()} on top of the stack");

    private List<object> PopToMark()
    {
        var at = _stack.LastIndexOf(Mark);
        if (at < 0) throw new InvalidDataException("the pickle uses a mark that was never set");
        var items = _stack.GetRange(at + 1, _stack.Count - at - 1);
        _stack.RemoveRange(at, _stack.Count - at);
        return items;
    }

    private object[] TupleFromMark() => PopToMark().ToArray();

    private object[] TupleOf(int n)
    {
        if (_stack.Count < n) throw new InvalidDataException("the pickle builds a tuple from too few items");
        var items = _stack.GetRange(_stack.Count - n, n).ToArray();
        _stack.RemoveRange(_stack.Count - n, n);
        return items;
    }

    private void Memoize(int index)
    {
        if (_stack.Count == 0) throw new InvalidDataException("the pickle memoizes an empty stack");
        if (_memo.Count >= MaxMemo) throw new InvalidDataException("the pickle memo is too large");
        _memo[index] = _stack[^1];
    }

    private object Recall(int index) =>
        _memo.TryGetValue(index, out var v) ? v : throw new InvalidDataException($"the pickle reads memo entry {index}, which was never stored");

    // ---- bytes ---------------------------------------------------------------------------------------------------------------------------

    private byte Byte()
    {
        if (_p >= _data.Length) throw new InvalidDataException("the pickle ends without a STOP (truncated)");
        return _data[_p++];
    }

    private ReadOnlySpan<byte> Take(int n)
    {
        if (n < 0 || _p + n > _data.Length) throw new InvalidDataException("the pickle ends inside a value (truncated)");
        var s = _data.AsSpan(_p, n);
        _p += n;
        return s;
    }

    private int UInt16() => BinaryPrimitives.ReadUInt16LittleEndian(Take(2));

    private int Int32() => BinaryPrimitives.ReadInt32LittleEndian(Take(4));

    private string Unicode()
    {
        var n = Int32();
        return Encoding.UTF8.GetString(Take(n));
    }

    private string Line()
    {
        var start = _p;
        while (_p < _data.Length && _data[_p] != (byte)'\n') _p++;
        if (_p >= _data.Length) throw new InvalidDataException("the pickle ends inside a name (truncated)");
        var s = Encoding.UTF8.GetString(_data, start, _p - start);
        _p++;
        return s;
    }

    private long Long1()
    {
        var n = Byte();
        if (n > 8) throw new InvalidDataException("the pickle has an integer wider than 64 bits");
        var bytes = Take(n);
        long v = 0;
        for (var i = 0; i < n; i++) v |= (long)bytes[i] << (8 * i);
        if (n > 0 && n < 8 && (bytes[n - 1] & 0x80) != 0) v |= -1L << (8 * n); // sign extension
        return v;
    }
}
