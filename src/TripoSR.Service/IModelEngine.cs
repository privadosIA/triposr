using TripoSR.Mesh;

namespace TripoSR.Service;

/// <summary>A model the engine made: the binary glTF and what is said of it in the job.</summary>
/// <param name="Glb">The <c>.glb</c>: upright (Y up), its origin at the middle of its base, with its colours, within <see cref="EngineInfo.MaxTriangles"/>.</param>
/// <param name="DetailUsed">The detail it was made at (lower than asked when the first had too many triangles).</param>
/// <param name="Size">The size that detail meant (<see cref="EngineInfo.DetailSizes"/>).</param>
/// <param name="Triangles">The triangles of the mesh.</param>
internal sealed record GenerationResult(byte[] Glb, string DetailUsed, int Size, int Triangles);

/// <summary>The engine of the service: one picture in, one model out.</summary>
internal interface IModelEngine : IDisposable
{
    /// <summary>What a model made by this engine says made it.</summary>
    string Version { get; }

    /// <summary>Makes the model of a picture (a png, jpg or webp). Throws <see cref="RefusedException"/> for what it will not make, with a sentence for the caller.</summary>
    GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel);
}

/// <summary>The engine refuses the request (not an error of the service): the message is what the caller is told.</summary>
internal sealed class RefusedException(string message) : Exception(message);

/// <summary>
/// The engine of the service: the TripoSR port. The model is loaded when the first job asks for it and stays in memory (on the card, with a cuda device) until the service stops:
/// there is one engine, so there is never another to free first.
/// </summary>
internal sealed class TripoSREngine(ServiceSettings settings) : IModelEngine
{
    private TripoSRPipeline? _pipeline;

    public string Version => EngineInfo.Version;

    public GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel)
    {
        try
        {
            _pipeline ??= TripoSRPipeline.Load(settings.CheckpointDirectory, settings.Device);
            var made = _pipeline.Make(picture, detail, cancel);
            return new GenerationResult(GlbWriter.Write(made.Mesh), made.DetailUsed, made.Size, made.Mesh.TriangleCount);
        }
        catch (ModelRefusedException e) { throw new RefusedException(e.Message); }
        catch (FileNotFoundException e) { throw new RefusedException(e.Message); } // a file of the weights is not there
    }

    public void Dispose() => _pipeline?.Dispose();
}

/// <summary>Chooses the engine from the settings.</summary>
internal static class EngineFactory
{
    public static IModelEngine Create(ServiceSettings settings) => new TripoSREngine(settings);
}
