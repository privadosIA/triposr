using TorchSharp;
using TripoSR.Field;
using TripoSR.Transformer;
using TripoSR.Triplane;
using TripoSR.Vision;
using static TorchSharp.torch;

namespace TripoSR;

#pragma warning disable IDE1006 // naming: the names are the checkpoint's keys

/// <summary>
/// <c>TSR</c> (the network of TripoSR, inference only): image tokens, triplane tokens, the backbone that crosses them, the upsampling to scene codes and the field
/// that is asked about points. Its children carry the names of the checkpoint (<c>image_tokenizer</c>, <c>tokenizer</c>, <c>backbone</c>, <c>post_processor</c>,
/// <c>decoder</c>), so the whole <c>model.ckpt</c> loads strictly with no mapping: every key used once, none left over.
/// </summary>
public sealed class TripoSRModel : nn.Module
{
    private readonly DinoSingleImageTokenizer image_tokenizer;
    private readonly TriplaneTokenizer tokenizer;
    private readonly Transformer1D backbone;
    private readonly TriplaneUpsampleNetwork post_processor;
    private readonly NeRFMLP decoder;

    public TripoSRConfig Config { get; }

    public TripoSRModel(TripoSRConfig? config = null, DinoConfig? dino = null) : base(nameof(TripoSRModel))
    {
        Config = config ?? new TripoSRConfig();
        image_tokenizer = new DinoSingleImageTokenizer(dino);
        tokenizer = new TriplaneTokenizer(Config);
        backbone = new Transformer1D(Config);
        post_processor = new TriplaneUpsampleNetwork(Config);
        decoder = new NeRFMLP(Config);
        RegisterComponents();
    }

    /// <summary>The field, to be asked about points (<see cref="TriplaneField"/>).</summary>
    public NeRFMLP Decoder => decoder;

    /// <summary>
    /// <c>TSR.forward</c>: the picture the model sees, (1, 512, 512, 3) in 0..1 (height, width, channel), to the scene codes (1, 3, 40, 64, 64).
    /// </summary>
    public Tensor SceneCodes(Tensor rgbCond)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope(); // everything but the result is freed here, not whenever the garbage collector gets to it
        var images = rgbCond.permute(0, 3, 1, 2).contiguous();   // (B, 3, H, W)
        var imageTokens = image_tokenizer.forward(images);        // (B, 1 + 32*32, 768)
        var tokens = tokenizer.forward((int)rgbCond.shape[0]);    // (B, 1024, 3 * 32 * 32)
        var backboneOut = backbone.forward(tokens, imageTokens);  // (B, 1024, 3072)
        var planes = tokenizer.detokenize(backboneOut);           // (B, 3, 1024, 32, 32)
        return post_processor.forward(planes).MoveToOuterDisposeScope();
    }
}
