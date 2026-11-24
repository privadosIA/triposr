using TorchSharp;
using static TorchSharp.torch;

namespace TripoSR.Picture;

/// <summary>From the cut-out to what the model sees: <c>resize_foreground(0.85)</c>, the composite over grey 0.5 and the antialiased bilinear resize to 512.</summary>
public static class ModelInput
{
    public const int Size = 512;
    public const float ForegroundRatio = 0.85f;

    /// <summary>
    /// <c>tsr.utils.resize_foreground</c>: the box of the pixels with alpha above 0 (its last row and column left out, as numpy's slice does) padded to a square and
    /// then to <c>size / ratio</c> with transparent pixels, centred.
    /// </summary>
    /// <exception cref="InvalidOperationException">The picture has no pixel with alpha above 0.</exception>
    public static RasterImage ResizeForeground(RasterImage rgba, double ratio)
    {
        int y1 = int.MaxValue, y2 = -1, x1 = int.MaxValue, x2 = -1;
        for (var y = 0; y < rgba.Height; y++)
            for (var x = 0; x < rgba.Width; x++)
                if (rgba[x, y, 3] > 0)
                {
                    if (y < y1) y1 = y;
                    if (y > y2) y2 = y;
                    if (x < x1) x1 = x;
                    if (x > x2) x2 = x;
                }
        if (y2 < 0) throw new InvalidOperationException("the picture has no object: everything in it is transparent");
        int fh = y2 - y1, fw = x2 - x1; // image[y1:y2, x1:x2]
        var size = Math.Max(fh, fw);
        int ph0 = (size - fh) / 2, pw0 = (size - fw) / 2;
        var newSize = (int)(size / ratio);
        var off = (newSize - size) / 2;
        var o = new RasterImage(newSize, newSize, 4);
        for (var y = 0; y < fh; y++)
            for (var x = 0; x < fw; x++)
                for (var c = 0; c < 4; c++)
                    o.Pixels[((y + ph0 + off) * newSize + x + pw0 + off) * 4 + c] = rgba.Pixels[((y1 + y) * rgba.Width + x1 + x) * 4 + c];
        return o;
    }

    /// <summary>The object over grey 0.5, as <c>server.py</c> does it (float32, truncated to 8 bits).</summary>
    public static RasterImage OverGrey(RasterImage rgba)
    {
        var o = new RasterImage(rgba.Width, rgba.Height, 3);
        for (int i = 0, n = rgba.Width * rgba.Height; i < n; i++)
        {
            var a = rgba.Pixels[i * 4 + 3] / 255.0f;
            for (var c = 0; c < 3; c++)
            {
                var v = rgba.Pixels[i * 4 + c] / 255.0f;
                var r = v * a + (1 - a) * 0.5f;
                o.Pixels[i * 3 + c] = (byte)(r * 255f);
            }
        }
        return o;
    }

    /// <summary>The picture the model sees before the resize: framed and over grey (RGB).</summary>
    public static RasterImage Prepare(RasterImage cutout) => OverGrey(ResizeForeground(cutout, ForegroundRatio));

    /// <summary>
    /// <c>ImagePreprocessor</c>: the 8-bit picture as floats in 0..1 resized to 512x512 with torch's bilinear filter with antialiasing; the result is
    /// (1, 512, 512, 3), height, width, channel (what <c>TSR.forward</c> calls <c>rgb_cond</c>).
    /// </summary>
    public static Tensor Resize(RasterImage rgb)
    {
        var src = new float[rgb.Pixels.Length];
        for (var i = 0; i < src.Length; i++) src[i] = rgb.Pixels[i] / 255.0f;
        var (hw, hb) = AntialiasedWeights(rgb.Width, Size);
        var (vw, vb) = AntialiasedWeights(rgb.Height, Size);
        // horizontal pass
        var mid = new float[rgb.Height * Size * 3];
        for (var y = 0; y < rgb.Height; y++)
            for (var xx = 0; xx < Size; xx++)
            {
                var (xmin, n) = hb[xx];
                for (var c = 0; c < 3; c++)
                {
                    float s = 0;
                    for (var k = 0; k < n; k++) s += src[(y * rgb.Width + xmin + k) * 3 + c] * hw[xx][k];
                    mid[(y * Size + xx) * 3 + c] = s;
                }
            }
        var dst = new float[Size * Size * 3];
        for (var yy = 0; yy < Size; yy++)
        {
            var (ymin, n) = vb[yy];
            for (var x = 0; x < Size; x++)
                for (var c = 0; c < 3; c++)
                {
                    float s = 0;
                    for (var k = 0; k < n; k++) s += mid[((ymin + k) * Size + x) * 3 + c] * vw[yy][k];
                    dst[(yy * Size + x) * 3 + c] = s;
                }
        }
        return tensor(dst, [1, Size, Size, 3]);
    }

    /// <summary>The weights of <c>upsample_bilinear2d</c> with <c>antialias=True</c> (a triangle filter as wide as the scale when shrinking), per output index.</summary>
    private static (float[][] weights, (int start, int count)[] bounds) AntialiasedWeights(int inSize, int outSize)
    {
        var scale = (double)inSize / outSize; // align_corners=False
        var support = scale >= 1.0 ? scale : 1.0;
        var invScale = scale >= 1.0 ? 1.0 / scale : 1.0;
        var weights = new float[outSize][];
        var bounds = new (int, int)[outSize];
        for (var i = 0; i < outSize; i++)
        {
            var center = scale * (i + 0.5);
            var xmin = Math.Max((int)(center - support + 0.5), 0);
            var xsize = Math.Min((int)(center + support + 0.5), inSize) - xmin;
            var w = new float[xsize];
            double total = 0;
            for (var j = 0; j < xsize; j++)
            {
                var x = (j + xmin - center + 0.5) * invScale;
                var v = Math.Abs(x) < 1.0 ? 1.0 - Math.Abs(x) : 0.0;
                w[j] = (float)v;
                total += v;
            }
            if (total != 0) for (var j = 0; j < xsize; j++) w[j] = (float)(w[j] / total);
            weights[i] = w;
            bounds[i] = (xmin, xsize);
        }
        return (weights, bounds);
    }
}
