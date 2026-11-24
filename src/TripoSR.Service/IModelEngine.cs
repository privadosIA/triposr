namespace TripoSR.Service;

/// <summary>A model the engine made: the binary glTF and what is said of it in the job.</summary>
/// <param name="Glb">The <c>.glb</c>: upright (Y up), its origin at the middle of its base, with its colours, within <see cref="EngineInfo.MaxTriangles"/>.</param>
/// <param name="DetailUsed">The detail it was made at (lower than asked when the first had too many triangles).</param>
/// <param name="Size">The size that detail meant (<see cref="EngineInfo.DetailSizes"/>).</param>
/// <param name="Triangles">The triangles of the mesh.</param>
internal sealed record GenerationResult(byte[] Glb, string DetailUsed, int Size, int Triangles);

/// <summary>The engine of the service: one picture in, one model out. This is where the port of TripoSR goes in.</summary>
internal interface IModelEngine : IDisposable
{
    /// <summary>What a model made by this engine says made it.</summary>
    string Version { get; }

    /// <summary>Makes the model of a picture (a png, jpg or webp). Throws <see cref="RefusedException"/> for what it will not make, with a sentence for the caller.</summary>
    GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel);
}

/// <summary>The engine refuses the request (not an error of the service): the message is what the caller is told.</summary>
internal sealed class RefusedException(string message) : Exception(message);

/// <summary>The engine of this build: the port is not done yet, so every model fails with a sentence that says so. It lets the service answer the whole contract from the first day.</summary>
internal sealed class NotPortedEngine : IModelEngine
{
    public string Version => EngineInfo.Version;

    public GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel) =>
        throw new RefusedException("TripoSR is not ported yet in this build: see openspec/changes/port-triposr");

    public void Dispose() { }
}

/// <summary>Chooses the engine from the settings. Until the port exists there is only <see cref="NotPortedEngine"/>.</summary>
internal static class EngineFactory
{
    public static IModelEngine Create(ServiceSettings settings) => new NotPortedEngine();
}
