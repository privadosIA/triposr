using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TripoSR.Picture;

/// <summary>
/// The cut-out of the reference (<c>rembg.remove(img.convert("RGB"), session=new_session("u2net"))</c> with its defaults): u2net on ONNX Runtime over the
/// picture at 320x320, the mask stretched to the minimum and the maximum, resized back with Lanczos, and the picture pasted over transparent with it.
/// A picture that already has transparency is used as it is.
/// </summary>
public sealed class U2NetCutout : IDisposable
{
    private const int Side = 320;
    private static readonly double[] Mean = [0.485, 0.456, 0.406];
    private static readonly double[] Std = [0.229, 0.224, 0.225];

    private readonly InferenceSession _session;
    private readonly string _input;

    public U2NetCutout(string onnxPath)
    {
        if (!File.Exists(onnxPath)) throw new FileNotFoundException($"The cut-out model was not found at {onnxPath}", onnxPath);
        _session = new InferenceSession(onnxPath);
        _input = _session.InputNames[0];
    }

    /// <summary>The picture with its background cut out (RGBA), or itself when it already has transparency.</summary>
    public RasterImage Cutout(RasterImage rgba)
    {
        if (HasTransparency(rgba)) return rgba;
        var rgb = ToRgb(rgba);
        var mask = Mask(rgb);
        return Paste(rgb, mask);
    }

    /// <summary>Decodes and cuts out in one step.</summary>
    public RasterImage Cutout(ReadOnlySpan<byte> picture) => Cutout(PictureDecoder.Decode(picture));

    /// <summary>The reference uses a picture as it is when its alpha reaches below 250 somewhere.</summary>
    public static bool HasTransparency(RasterImage rgba)
    {
        for (var i = 3; i < rgba.Pixels.Length; i += 4)
            if (rgba.Pixels[i] < 250) return true;
        return false;
    }

    private static RasterImage ToRgb(RasterImage rgba)
    {
        var rgb = new RasterImage(rgba.Width, rgba.Height, 3);
        for (int i = 0, n = rgba.Width * rgba.Height; i < n; i++)
        {
            rgb.Pixels[i * 3] = rgba.Pixels[i * 4];
            rgb.Pixels[i * 3 + 1] = rgba.Pixels[i * 4 + 1];
            rgb.Pixels[i * 3 + 2] = rgba.Pixels[i * 4 + 2];
        }
        return rgb;
    }

    /// <summary>The mask (one byte per pixel, the picture's size) of the object that u2net sees.</summary>
    public byte[] Mask(RasterImage rgb)
    {
        var small = PilResample.Resize(rgb.Pixels, rgb.Width, rgb.Height, 3, Side, Side);
        var max = 0;
        foreach (var b in small) if (b > max) max = b;
        var divisor = Math.Max((double)max, 1e-6);
        var input = new DenseTensor<float>([1, 3, Side, Side]);
        for (var y = 0; y < Side; y++)
            for (var x = 0; x < Side; x++)
                for (var c = 0; c < 3; c++)
                    input[0, c, y, x] = (float)((small[(y * Side + x) * 3 + c] / divisor - Mean[c]) / Std[c]);

        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_input, input)]);
        var pred = results.First().AsTensor<float>().ToArray(); // [1, 1, 320, 320]; the reference takes pred[:, 0]
        var mi = pred.Min();
        var ma = pred.Max();
        var small8 = new byte[Side * Side];
        for (var i = 0; i < small8.Length; i++)
        {
            var v = (pred[i] - mi) / (ma - mi);
            v = Math.Clamp(v, 0f, 1f);
            small8[i] = (byte)(v * 255f);
        }
        return PilResample.Resize(small8, Side, Side, 1, rgb.Width, rgb.Height);
    }

    /// <summary>Pillow's <c>Image.composite(rgb, transparent, mask)</c>: every channel, the alpha included, is the picture's (alpha 255) times the mask.</summary>
    public static RasterImage Paste(RasterImage rgb, byte[] mask)
    {
        var o = new RasterImage(rgb.Width, rgb.Height, 4);
        for (int i = 0, n = rgb.Width * rgb.Height; i < n; i++)
        {
            for (var c = 0; c < 3; c++) o.Pixels[i * 4 + c] = Blend(mask[i], rgb.Pixels[i * 3 + c]);
            o.Pixels[i * 4 + 3] = Blend(mask[i], 255);
        }
        return o;
    }

    // Pillow's BLEND(mask, 0, value): DIV255(value * mask) with its rounding
    private static byte Blend(int mask, int value)
    {
        var t = value * mask + 128;
        return (byte)(((t >> 8) + t) >> 8);
    }

    public void Dispose() => _session.Dispose();
}
