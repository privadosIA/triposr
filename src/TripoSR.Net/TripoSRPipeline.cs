using TorchSharp;
using TripoSR.Mesh;
using TripoSR.Picture;
using TripoSR.Vision;
using TripoSR.Weights;
using static TorchSharp.torch;

namespace TripoSR;

/// <summary>
/// One picture in, one model out: what the container's <c>run_job</c> does with the TripoSR engine. The weights are read from a folder with the layout of the container's
/// <c>/ckpt</c> (<see cref="EngineInfo.RequiredFiles"/>); nothing is downloaded and no Python is used. Not safe for two models at once (the service makes one at a time).
/// </summary>
public sealed class TripoSRPipeline : IDisposable
{
    private readonly TripoSRModel _model;
    private readonly U2NetCutout _cutout;
    private readonly Device _device;

    public string Device => _device.ToString();

    private TripoSRPipeline(TripoSRModel model, U2NetCutout cutout, Device device) => (_model, _cutout, _device) = (model, cutout, device);

    /// <summary>
    /// Reads the weights and builds the network on <paramref name="device"/> (<c>cpu</c>, <c>cuda</c>, <c>cuda:1</c>). The checkpoint is loaded strictly: a key that is missing, left over
    /// or of another shape is named in the error, and nothing is loaded.
    /// </summary>
    /// <exception cref="FileNotFoundException">A file of the weights folder is missing.</exception>
    /// <exception cref="InvalidDataException">The files are not the TripoSR network of this port.</exception>
    public static TripoSRPipeline Load(string weightsDir, string device = "cpu")
    {
        foreach (var f in EngineInfo.RequiredFiles)
            if (!File.Exists(Path.Combine(weightsDir, f))) throw new FileNotFoundException($"The weights are not complete: {f} is missing from {weightsDir}", Path.Combine(weightsDir, f));
        var config = new TripoSRConfig();
        config.Verify(File.ReadAllText(Path.Combine(weightsDir, "triposr", "config.yaml")), "triposr/config.yaml");
        var dev = new Device(device);
        if (dev.type == DeviceType.CUDA && !cuda.is_available()) throw new InvalidOperationException($"the device {device} was asked for but this build has no CUDA (build with -p:TorchBackend=cuda-linux) or there is no card");

        var model = new TripoSRModel(config, DinoConfig.FromFile(Path.Combine(weightsDir, "triposr", "dino-vitb16-config.json")));
        try
        {
            var checkpoint = TorchCheckpoint.Open(Path.Combine(weightsDir, "triposr", "model.ckpt"));
            TorchCheckpointLoader.Load(model, checkpoint, dev, "", "The checkpoint does not match the TripoSR network:", "checkpoint keys");
            model.eval();
            var cutout = new U2NetCutout(Path.Combine(weightsDir, "rembg", "u2net.onnx"));
            return new TripoSRPipeline(model, cutout, dev);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>The picture with its background cut out (a picture that has transparency is used as it is).</summary>
    public RasterImage Cutout(byte[] picture) => _cutout.Cutout(picture);

    /// <summary>The scene codes of a cut-out picture: (1, 3, 40, 64, 64) on the device.</summary>
    public Tensor SceneCodes(RasterImage cutout)
    {
        using var _ = no_grad();
        using var rgbCond = ModelInput.Resize(ModelInput.Prepare(cutout)).to(_device);
        return _model.SceneCodes(rgbCond);
    }

    /// <summary>The model of a picture (png, jpg or webp) at <paramref name="detail"/> (<c>low</c>, <c>normal</c>, <c>high</c>), finished as the container finishes it.</summary>
    /// <exception cref="ModelRefusedException">Not a picture, no object in it, no surface or too many triangles at every detail: with the sentence for the caller.</exception>
    public FinishedModel Make(byte[] picture, string detail, CancellationToken cancel = default)
    {
        ModelFinish.DetailsFrom(detail); // refuses a detail that does not exist before any work is done
        RasterImage cutout;
        try { cutout = Cutout(picture); }
        catch (InvalidDataException e) { throw new ModelRefusedException(e.Message); }
        cancel.ThrowIfCancellationRequested();
        Tensor codes;
        try { codes = SceneCodes(cutout); }
        catch (InvalidOperationException e) when (e.Message.StartsWith("the picture has no object")) { throw new ModelRefusedException(e.Message); }
        using (codes)
        {
            using var sceneCode = codes[0];
            return ModelFinish.Make(detail, size =>
            {
                cancel.ThrowIfCancellationRequested();
                return SurfaceExtractor.Extract(_model.Decoder, sceneCode, size, _model.Config);
            });
        }
    }

    public void Dispose()
    {
        _cutout.Dispose();
        _model.Dispose();
    }
}
