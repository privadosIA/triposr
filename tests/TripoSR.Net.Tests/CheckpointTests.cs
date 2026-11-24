using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using Xunit.Abstractions;

namespace TripoSR.Net.Tests;

/// <summary>
/// The checkpoint of the container (<c>triposr/model.ckpt</c>, a PyTorch pickle) read in C# without Python, against what the Python reference says
/// it holds: <c>ckpt_keys.json</c> of the golden run (name, shape, dtype and sha256 of the bytes of every tensor).
/// </summary>
public sealed class CheckpointTests(ITestOutputHelper output)
{
    private sealed record Key(string name, long[] shape, string dtype, string sha256);

    private static Key[] ReferenceKeys() =>
        JsonSerializer.Deserialize<Key[]>(Fixtures.ReadAllText("ckpt_keys.json"))!;

    [Fact]
    public void TheReferenceKeysAreThoseOfTheModel()
    {
        // the five groups of the module tree (the names the C# modules replicate), 549 float32 tensors
        var keys = ReferenceKeys();
        Assert.Equal(549, keys.Length);
        Assert.All(keys, k => Assert.Equal("float32", k.dtype));
        var groups = keys.GroupBy(k => k.name.Split('.')[0]).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(new Dictionary<string, int> { ["image_tokenizer"] = 200, ["tokenizer"] = 1, ["backbone"] = 326, ["post_processor"] = 2, ["decoder"] = 20 }, groups);
    }

    [Fact]
    public void TheEngineNeedsTheFilesOfTheContainer()
    {
        // docker/model3d/server.py: REQUIRED["triposr"]
        Assert.Equal(new[] { "triposr/config.yaml", "triposr/model.ckpt", "triposr/dino-vitb16-config.json", "rembg/u2net.onnx" }, EngineInfo.RequiredFiles);
    }

    [RequiresCheckpoint]
    public void TheWeightsFolderHasEveryRequiredFile()
    {
        Assert.All(EngineInfo.RequiredFiles, f => Assert.True(File.Exists(Path.Combine(Checkpoint.Dir!, f)), f));
    }

    [RequiresCheckpoint]
    public void EveryKeyIsInTheFileAndTheOtherWayRound()
    {
        var keys = ReferenceKeys();
        var file = Checkpoint.Model.Tensors;
        Assert.Empty(keys.Select(k => k.name).Except(file.Keys).Take(5));   // a key of the reference the reader did not find
        Assert.Empty(file.Keys.Except(keys.Select(k => k.name)).Take(5));   // a key the reader found and the reference does not have
        foreach (var k in keys)
        {
            var t = file[k.name];
            Assert.Equal("F32", t.Dtype);
            Assert.True(k.shape.SequenceEqual(t.Shape), $"{k.name}: shape [{string.Join(",", t.Shape)}], the reference says [{string.Join(",", k.shape)}]");
        }
    }

    [RequiresCheckpoint]
    public void EveryTensorIsReadBitForBit()
    {
        var keys = ReferenceKeys();
        var bad = new List<string>();
        foreach (var k in keys)
        {
            using var t = Checkpoint.Model.Read(k.name);
            var hash = Convert.ToHexString(SHA256.HashData(t.bytes)).ToLowerInvariant();
            if (hash != k.sha256) bad.Add(k.name);
        }
        output.WriteLine($"{keys.Length - bad.Count}/{keys.Length} tensors identical to the reference's bytes");
        Assert.True(bad.Count == 0, "different bytes: " + string.Join(", ", bad.Take(10)));
    }

    [RequiresCheckpoint]
    public void ASafetensorsFileHasTheSameTensors()
    {
        var path = Path.Combine(Path.GetTempPath(), "triposr-ckpt-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            var tensors = Checkpoint.Model.Tensors.Keys.ToDictionary(n => n, n => Checkpoint.Model.Read(n));
            MusicGen.Core.Weights.SafeTensorsWriter.Save(path, tensors);
            foreach (var t in tensors.Values) t.Dispose();
            var back = MusicGen.Core.Weights.CheckpointIndex.Open(path);
            Assert.Equal(Checkpoint.Model.Tensors.Count, back.Tensors.Count);
            foreach (var k in ReferenceKeys().Take(20))
            {
                using var t = back.Read(k.name);
                Assert.Equal(k.sha256, Convert.ToHexString(SHA256.HashData(t.bytes)).ToLowerInvariant());
            }
        }
        finally { File.Delete(path); }
    }
}
