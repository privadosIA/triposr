using MusicGen.Core.Testing;
using TripoSR.Picture;
using Xunit.Abstractions;

namespace TripoSR.Net.Tests;

/// <summary>The picture stage against the golden run: the cut-out, the framing over grey and the 512x512 input of the model.</summary>
public sealed class PictureTests(ITestOutputHelper output)
{
    public static IEnumerable<object[]> WithAlpha() => new[] { "chair", "hamburger", "horse" }.Select(n => new object[] { n });
    public static IEnumerable<object[]> WithoutAlpha() => new[] { "teapot", "poly_fox" }.Select(n => new object[] { n });
    public static IEnumerable<object[]> All() => WithAlpha().Concat(WithoutAlpha());

    private static string Picture(string name) => Directory.GetFiles(Path.Combine(Fixtures.Dir, "pictures"), name + ".*").Single();

    private static RasterImage Decode(string path) => PictureDecoder.Decode(File.ReadAllBytes(path));

    private static RasterImage Rgb(RasterImage rgba)
    {
        var o = new RasterImage(rgba.Width, rgba.Height, 3);
        for (int i = 0, n = rgba.Width * rgba.Height; i < n; i++) Array.Copy(rgba.Pixels, i * 4, o.Pixels, i * 3, 3);
        return o;
    }

    private static (int differing, int maxDiff, double mean) Diff(byte[] a, byte[] b)
    {
        Assert.Equal(a.Length, b.Length);
        int differing = 0, max = 0;
        long sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Math.Abs(a[i] - b[i]);
            if (d != 0) differing++;
            if (d > max) max = d;
            sum += d;
        }
        return (differing, max, (double)sum / a.Length);
    }

    [Theory, MemberData(nameof(WithAlpha))]
    public void APictureWithTransparencyIsUsedAsItIs(string name)
    {
        var picture = Decode(Picture(name));
        Assert.True(U2NetCutout.HasTransparency(picture));
        var cutout = Decode(Path.Combine(Fixtures.Dir, name + ".cutout.png"));
        Assert.Equal(cutout.Pixels, picture.Pixels);
    }

    [Theory, MemberData(nameof(WithoutAlpha))]
    public void APictureWithNoTransparencyHasNone(string name) =>
        Assert.False(U2NetCutout.HasTransparency(Decode(Picture(name))));

    [RequiresCheckpointTheory, MemberData(nameof(WithoutAlpha))]
    public void TheCutoutMatchesRembg(string name)
    {
        using var u2net = new U2NetCutout(Path.Combine(Checkpoint.Dir!, "rembg", "u2net.onnx"));
        var got = u2net.Cutout(File.ReadAllBytes(Picture(name)));
        var expected = Decode(Path.Combine(Fixtures.Dir, name + ".cutout.png"));
        Assert.Equal((expected.Width, expected.Height), (got.Width, got.Height));
        var (differing, max, mean) = Diff(got.Pixels, expected.Pixels);
        output.WriteLine($"{name}: {differing}/{got.Pixels.Length} bytes differ, max {max}, mean {mean:F5}");
        Assert.True(max <= 1 && mean < 0.001, $"{name}: max {max}, mean {mean}");
    }

    [Theory, MemberData(nameof(All))]
    public void TheFramingOverGreyIsThePictureOfTheReference(string name)
    {
        var cutout = Decode(Path.Combine(Fixtures.Dir, name + ".cutout.png"));
        var got = ModelInput.Prepare(cutout);
        var expected = Decode(Path.Combine(Fixtures.Dir, name + ".input.png"));
        Assert.Equal((expected.Width, expected.Height), (got.Width, got.Height));
        Assert.Equal(Rgb(expected).Pixels, got.Pixels);
    }

    [Theory, MemberData(nameof(All))]
    public void TheModelInputIsWhatTheReferenceFeeds(string name)
    {
        var input = Rgb(Decode(Path.Combine(Fixtures.Dir, name + ".input.png")));
        using var got = ModelInput.Resize(input);
        using var expected = Fixtures.Load(name + ".stages.safetensors")["rgb_cond"];
        Assert.Equal(expected.shape, got.shape);
        var r = Parity.Compare(got, expected);
        output.WriteLine($"{name}: {r}");
        Assert.True(r.MaxAbs <= Stage.MaxAbsLimit(expected) && r.CosSim >= Stage.MinCosSim, r.ToString());
    }

    [Fact]
    public void ABodyThatIsNotAPictureIsRefusedWithASentence()
    {
        var e = Assert.Throws<InvalidDataException>(() => PictureDecoder.Decode(new byte[] { 1, 2, 3, 4 }));
        Assert.Contains("not a picture", e.Message);
    }

    [Fact]
    public void ASolidTransparentPictureHasNoObject()
    {
        var e = Assert.Throws<InvalidOperationException>(() => ModelInput.ResizeForeground(new RasterImage(8, 8, 4), 0.85));
        Assert.Contains("no object", e.Message);
    }
}
