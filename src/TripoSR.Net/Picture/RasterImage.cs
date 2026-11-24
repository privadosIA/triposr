namespace TripoSR.Picture;

/// <summary>An 8-bit picture, interleaved (RGB or RGBA), top row first.</summary>
public sealed record RasterImage(int Width, int Height, int Channels, byte[] Pixels)
{
    public RasterImage(int width, int height, int channels) : this(width, height, channels, new byte[width * height * channels]) { }

    public byte this[int x, int y, int c] => Pixels[(y * Width + x) * Channels + c];
}
