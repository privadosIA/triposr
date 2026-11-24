using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

// Adapted from ~/Proyectos/lion-clap (LionClap.Weights): the keys of the checkpoint may carry a prefix the module does not (<c>stripPrefix</c>).
namespace TripoSR.Weights;

/// <summary>
/// Strict loading from a <see cref="TorchCheckpoint"/>, with the guarantees of musicgen-core's <c>StrictLoader</c> (which needs a safetensors index and cannot
/// be built from a PyTorch file): the keys and shapes decide from the headers alone, before anything is read; a mismatch is one
/// <see cref="InvalidDataException"/> that lists every discrepancy and leaves the model untouched; otherwise the tensors are read one at a time, in file
/// order, under <c>no_grad</c>, as float32 (the integer buffers round-trip exactly), and copied in. Parameters and buffers are both loaded.
/// </summary>
public static class TorchCheckpointLoader
{
    /// <param name="prefix">Only checkpoint keys that start with it belong to the model (<c>audio_</c>, <c>text_</c>); the others are never read.</param>
    /// <param name="errorPrefix">First line of the error.</param>
    /// <param name="unexpectedLabel">Wording of the "keys in the checkpoint that the model does not have" line.</param>
    /// <param name="stripPrefix">What the module's own names lack: <c>image_tokenizer.model.</c> turns the key <c>image_tokenizer.model.layernorm.weight</c> into <c>layernorm.weight</c>.</param>
    public static void Load(nn.Module model, TorchCheckpoint checkpoint, Device device, string prefix, string errorPrefix, string unexpectedLabel, string? stripPrefix = null)
    {
        var strip = (stripPrefix ?? "").Length;
        var entries = checkpoint.Tensors.Where(kv => kv.Key.StartsWith(prefix, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[strip..], kv => kv.Value, StringComparer.Ordinal);
        var state = State(model).ToDictionary(kv => kv.Key, kv => kv.Value.shape, StringComparer.Ordinal);

        var problems = new List<string>();
        var missing = state.Keys.Where(k => !entries.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
        var unexpected = entries.Keys.Where(k => !state.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
        var shapes = state.Keys.Where(entries.ContainsKey).Order(StringComparer.Ordinal)
            .Where(k => !state[k].AsSpan().SequenceEqual(entries[k].Shape))
            .Select(k => $"{k}: the model has [{string.Join(", ", state[k])}], the checkpoint [{string.Join(", ", entries[k].Shape)}]")
            .ToList();
        if (missing.Count > 0) problems.Add($"{missing.Count} model keys are missing from the checkpoint: {string.Join(", ", missing)}");
        if (unexpected.Count > 0) problems.Add($"{unexpected.Count} {unexpectedLabel} are not in the model: {string.Join(", ", unexpected)}");
        if (shapes.Count > 0) problems.Add($"shape mismatches ({shapes.Count}): {string.Join("; ", shapes)}");
        if (problems.Count > 0) throw new InvalidDataException(errorPrefix + "\n  " + string.Join("\n  ", problems));

        model.to(device);               // the tensors are replaced: take the state afterwards
        var target = State(model);
        using var _ = no_grad();
        foreach (var (key, _) in entries.OrderBy(kv => kv.Value.Offset))
        {
            using var w = checkpoint.Read((stripPrefix ?? "") + key, ScalarType.Float32);
            target[key].copy_(w);
        }
    }

    private static Dictionary<string, Tensor> State(nn.Module model)
    {
        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, p) in model.named_parameters()) state[name] = p;
        foreach (var (name, b) in model.named_buffers()) state[name] = b;
        return state;
    }
}
