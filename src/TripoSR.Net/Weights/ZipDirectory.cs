using System.Buffers.Binary;
using System.Text;

// Copied from ~/Proyectos/lion-clap (LionClap.Weights): to be moved to musicgen-core and shared by both (see docs/porting-plan.md).
namespace TripoSR.Weights;

/// <summary>An entry of a zip archive: its name, the absolute position of its (stored) data and its size.</summary>
internal sealed record ZipEntry(string Name, long DataOffset, long Size);

/// <summary>
/// The minimum of the zip format that a PyTorch checkpoint needs: the central directory and, for every entry, where its data starts.
/// Only stored (uncompressed) entries and plain zip (no zip64) are accepted; anything else is an error with the cause.
/// </summary>
internal static class ZipDirectory
{
    private const uint EndOfCentralDirectory = 0x06054b50, CentralHeader = 0x02014b50, LocalHeader = 0x04034b50;

    public static IReadOnlyList<ZipEntry> Read(Stream stream)
    {
        var length = stream.Length;
        if (length < 22) throw new InvalidDataException("not a readable PyTorch checkpoint: the file is too small to be a zip archive");

        // the end-of-central-directory record is the last 22 bytes unless a comment follows it (then it is within the last 22 + 65535)
        var tail = ReadAt(stream, length - 22, 22);
        var tailSize = 22;
        var at = BinaryPrimitives.ReadUInt32LittleEndian(tail) == EndOfCentralDirectory ? 0 : -1;
        if (at < 0)
        {
            tailSize = (int)Math.Min(length, 22 + 65535);
            tail = ReadAt(stream, length - tailSize, tailSize);
            for (var i = tailSize - 22; i >= 0; i--)
            {
                if (BinaryPrimitives.ReadUInt32LittleEndian(tail.AsSpan(i)) == EndOfCentralDirectory) { at = i; break; }
            }
        }
        if (at < 0) throw new InvalidDataException("not a readable PyTorch checkpoint: no zip end-of-central-directory record (truncated or not a zip)");

        var count = BinaryPrimitives.ReadUInt16LittleEndian(tail.AsSpan(at + 10));
        var directorySize = BinaryPrimitives.ReadUInt32LittleEndian(tail.AsSpan(at + 12));
        var directoryOffset = BinaryPrimitives.ReadUInt32LittleEndian(tail.AsSpan(at + 16));
        if (count == 0xFFFF || directorySize == 0xFFFFFFFF || directoryOffset == 0xFFFFFFFF)
            throw new NotSupportedException("zip64 archives are not supported");
        if (directoryOffset + (long)directorySize > length)
            throw new InvalidDataException("not a readable PyTorch checkpoint: the zip central directory lies outside the file (truncated)");

        var directory = ReadAt(stream, directoryOffset, (int)directorySize);
        var entries = new List<ZipEntry>(count);
        var p = 0;
        for (var n = 0; n < count; n++)
        {
            if (p + 46 > directory.Length || BinaryPrimitives.ReadUInt32LittleEndian(directory.AsSpan(p)) != CentralHeader)
                throw new InvalidDataException("not a readable PyTorch checkpoint: damaged zip central directory");
            var method = BinaryPrimitives.ReadUInt16LittleEndian(directory.AsSpan(p + 10));
            var compressed = BinaryPrimitives.ReadUInt32LittleEndian(directory.AsSpan(p + 20));
            var size = BinaryPrimitives.ReadUInt32LittleEndian(directory.AsSpan(p + 24));
            var nameLength = BinaryPrimitives.ReadUInt16LittleEndian(directory.AsSpan(p + 28));
            var extraLength = BinaryPrimitives.ReadUInt16LittleEndian(directory.AsSpan(p + 30));
            var commentLength = BinaryPrimitives.ReadUInt16LittleEndian(directory.AsSpan(p + 32));
            var localOffset = BinaryPrimitives.ReadUInt32LittleEndian(directory.AsSpan(p + 42));
            if (p + 46 + nameLength + extraLength + commentLength > directory.Length)
                throw new InvalidDataException("not a readable PyTorch checkpoint: damaged zip central directory");
            var name = Encoding.UTF8.GetString(directory, p + 46, nameLength);
            p += 46 + nameLength + extraLength + commentLength;

            if (compressed == 0xFFFFFFFF || size == 0xFFFFFFFF || localOffset == 0xFFFFFFFF)
                throw new NotSupportedException($"zip64 entry {name} is not supported");
            if (method != 0)
                throw new InvalidDataException($"entry {name} is compressed (zip method {method}); a PyTorch checkpoint stores its tensors uncompressed");
            if (compressed != size) throw new InvalidDataException($"entry {name} has inconsistent sizes");

            // the data follows the local header, whose name and extra lengths can differ from the central ones (torch pads with the extra field)
            var local = ReadAt(stream, localOffset, 30);
            if (BinaryPrimitives.ReadUInt32LittleEndian(local) != LocalHeader)
                throw new InvalidDataException($"entry {name}: bad local header (truncated or damaged zip)");
            var dataOffset = localOffset + 30L + BinaryPrimitives.ReadUInt16LittleEndian(local.AsSpan(26)) + BinaryPrimitives.ReadUInt16LittleEndian(local.AsSpan(28));
            if (dataOffset + size > length)
                throw new InvalidDataException($"entry {name} extends past the end of the file (truncated)");
            entries.Add(new ZipEntry(name, dataOffset, size));
        }
        return entries;
    }

    public static byte[] ReadAt(Stream stream, long offset, int count)
    {
        var buffer = new byte[count];
        stream.Seek(offset, SeekOrigin.Begin);
        var read = 0;
        while (read < count)
        {
            var n = stream.Read(buffer, read, count - read);
            if (n <= 0) throw new InvalidDataException("not a readable PyTorch checkpoint: unexpected end of file (truncated)");
            read += n;
        }
        return buffer;
    }
}
