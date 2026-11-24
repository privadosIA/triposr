using SkiaSharp;

namespace TripoSR.Picture;

/// <summary>Decodes a png, jpg or webp (what the service takes) into 8-bit RGBA, with no colour management, no premultiplying and no EXIF turning (as the reference's <c>Image.open</c>).</summary>
public static class PictureDecoder
{
    /// <exception cref="InvalidDataException">The bytes are not a picture of a supported kind.</exception>
    public static RasterImage Decode(ReadOnlySpan<byte> bytes)
    {
        using var data = SKData.CreateCopy(bytes.ToArray());
        using var codec = SKCodec.Create(data);
        if (codec is null) throw new InvalidDataException("the body is not a picture: png, jpg or webp are the formats taken");
        if (codec.EncodedFormat is not (SKEncodedImageFormat.Png or SKEncodedImageFormat.Jpeg or SKEncodedImageFormat.Webp))
            throw new InvalidDataException($"the picture is a {codec.EncodedFormat}: png, jpg or webp are the formats taken");
        var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var pixels = new byte[info.Width * info.Height * 4];
        unsafe
        {
            fixed (byte* p = pixels)
            {
                var result = codec.GetPixels(info, (IntPtr)p);
                if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
                    throw new InvalidDataException($"the picture cannot be decoded ({result})");
            }
        }
        return new RasterImage(info.Width, info.Height, 4, pixels);
    }
}
