namespace TripoSR.Picture;

/// <summary>
/// Pillow's <c>Image.resize</c> for 8-bit pictures, step by step: the same filter support, the same normalised coefficients turned into 22-bit
/// fixed point and the same rounding, a horizontal pass and then a vertical one. The cut-out of the reference (rembg) resizes with it, so the
/// port gets the same pixels, not just similar ones.
/// </summary>
public static class PilResample
{
    private const int PrecisionBits = 32 - 8 - 2;

    public enum Filter { Lanczos }

    private static double Sinc(double x) => x == 0.0 ? 1.0 : Math.Sin(x * Math.PI) / (x * Math.PI);

    private static double Lanczos3(double x) => x >= -3.0 && x < 3.0 ? Sinc(x) * Sinc(x / 3.0) : 0.0;

    /// <summary>Coefficients (fixed point) and first input index of each output pixel along one axis.</summary>
    private static (int ksize, int[] bounds, int[] kk) Coefficients(int inSize, int outSize)
    {
        const double support0 = 3.0;
        var scale = (double)inSize / outSize;
        var filterScale = Math.Max(scale, 1.0);
        var support = support0 * filterScale;
        var ksize = (int)Math.Ceiling(support) * 2 + 1;
        var kk = new int[outSize * ksize];
        var bounds = new int[outSize * 2];
        var k = new double[ksize];
        var ss = 1.0 / filterScale;
        for (var xx = 0; xx < outSize; xx++)
        {
            var center = (xx + 0.5) * scale;
            var xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            var xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            double ww = 0.0;
            Array.Clear(k);
            for (var x = 0; x < xmax; x++)
            {
                var w = Lanczos3((x + xmin - center + 0.5) * ss);
                k[x] = w;
                ww += w;
            }
            for (var x = 0; x < xmax; x++)
                if (ww != 0.0) k[x] /= ww;
            for (var x = 0; x < ksize; x++)
                kk[xx * ksize + x] = k[x] < 0 ? (int)(-0.5 + k[x] * (1 << PrecisionBits)) : (int)(0.5 + k[x] * (1 << PrecisionBits));
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return (ksize, bounds, kk);
    }

    private static byte Clip8(long v) => (byte)Math.Clamp(v >> PrecisionBits, 0, 255);

    /// <summary>Resizes an interleaved 8-bit picture (<paramref name="channels"/> bytes per pixel) with the Lanczos filter, as Pillow does.</summary>
    public static byte[] Resize(byte[] src, int width, int height, int channels, int newWidth, int newHeight)
    {
        if (newWidth == width && newHeight == height) return (byte[])src.Clone();
        var cur = src;
        var w = width;
        if (newWidth != width)
        {
            var (ksize, bounds, kk) = Coefficients(width, newWidth);
            var dst = new byte[newWidth * height * channels];
            for (var y = 0; y < height; y++)
                for (var xx = 0; xx < newWidth; xx++)
                {
                    var xmin = bounds[xx * 2];
                    var xmax = bounds[xx * 2 + 1];
                    for (var c = 0; c < channels; c++)
                    {
                        long ss = 1L << (PrecisionBits - 1);
                        for (var x = 0; x < xmax; x++) ss += (long)cur[(y * w + xmin + x) * channels + c] * kk[xx * ksize + x];
                        dst[(y * newWidth + xx) * channels + c] = Clip8(ss);
                    }
                }
            cur = dst;
            w = newWidth;
        }
        if (newHeight != height)
        {
            var (ksize, bounds, kk) = Coefficients(height, newHeight);
            var dst = new byte[w * newHeight * channels];
            for (var yy = 0; yy < newHeight; yy++)
            {
                var ymin = bounds[yy * 2];
                var ymax = bounds[yy * 2 + 1];
                for (var x = 0; x < w; x++)
                    for (var c = 0; c < channels; c++)
                    {
                        long ss = 1L << (PrecisionBits - 1);
                        for (var y = 0; y < ymax; y++) ss += (long)cur[((ymin + y) * w + x) * channels + c] * kk[yy * ksize + y];
                        dst[(yy * w + x) * channels + c] = Clip8(ss);
                    }
            }
            cur = dst;
        }
        return cur;
    }
}
