using System.Diagnostics;
using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using TripoSR.Service;
using Xunit.Abstractions;

namespace TripoSR.Net.Tests;

/// <summary>
/// The five pictures through the service with the real engine and the real weights (7.2): the same details as the container, and each model matching the container's in
/// triangles, bounding box and Chamfer distance. Skipped without <c>TRIPOSR_CKPT</c>; <c>TRIPOSR_DEVICE=cuda</c> (with a CUDA build) runs it on the card.
/// </summary>
public sealed class ServiceRealEngineTests(ITestOutputHelper output) : IAsyncLifetime
{
    private WebApplication? _app;
    private HttpClient _http = null!;

    public async Task InitializeAsync()
    {
        if (Checkpoint.Dir is null) return;
        var settings = new ServiceSettings(Checkpoint.Dir, "", 0, "test", Checkpoint.Env.DeviceName);
        var engine = EngineFactory.Create(settings);
        var gate = new FifoGate();
        var probe = new HealthProbe(settings, gate, engine.Version, TimeSpan.Zero);
        _app = ServiceApp.Build(new ServiceApp(settings, new JobStore(engine, gate, probe.MissingFiles), probe, gate), "http://127.0.0.1:0");
        await _app.StartAsync();
        var address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task DisposeAsync()
    {
        if (_app is null) return;
        _http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    public static IEnumerable<object[]> Cases() =>
        new[] { "chair", "hamburger", "horse", "teapot", "poly_fox" }.SelectMany(n => new[] { "low", "normal", "high" }.Select(d => new object[] { n, d }));

    private async Task<JsonElement> Make(string name, string detail, Stopwatch clock)
    {
        var picture = Directory.GetFiles(Path.Combine(Fixtures.Dir, "pictures"), name + ".*").Single();
        var r = await _http.PostAsync($"/generate?engine=triposr&detail={detail}", new ByteArrayContent(await File.ReadAllBytesAsync(picture)));
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var id = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString()!;
        clock.Restart();
        while (true)
        {
            var job = JsonDocument.Parse(await _http.GetStringAsync($"/jobs/{id}")).RootElement.Clone();
            if (job.GetProperty("state").GetString() is "done" or "failed") return job;
            await Task.Delay(50);
        }
    }

    [RequiresCheckpointTheory, MemberData(nameof(Cases))]
    public async Task EveryPictureAtEveryDetailIsTheModelOfTheContainer(string name, string detail)
    {
        var clock = new Stopwatch();
        var job = await Make(name, detail, clock);
        Assert.True(job.GetProperty("state").GetString() == "done", job.GetProperty("error").GetString());
        using var meta = JsonDocument.Parse(Fixtures.ReadAllText(name + ".json"));
        var expected = meta.RootElement.GetProperty("models").GetProperty(detail);
        Assert.Equal(expected.GetProperty("detail_used").GetString(), job.GetProperty("detail_used").GetString());
        Assert.Equal(EngineInfo.DetailSizes[expected.GetProperty("detail_used").GetString()!], job.GetProperty("resolution").GetInt32());

        var id = job.GetProperty("id").GetString()!;
        var glb = await _http.GetByteArrayAsync($"/jobs/{id}/model");
        var (got, triangles) = MeshCompare.ReadGlb(glb);
        var reference = MeshCompare.ReadGlb(File.ReadAllBytes(Fixtures.PathOf($"{name}.{detail}.glb")));
        Assert.Equal(triangles, job.GetProperty("triangles").GetInt32());

        var (glo, ghi) = MeshCompare.Bounds(got);
        var (rlo, rhi) = MeshCompare.Bounds(reference.Vertices);
        var size = Math.Sqrt(Enumerable.Range(0, 3).Sum(k => (rhi[k] - rlo[k]) * (rhi[k] - rlo[k])));
        var box = Enumerable.Range(0, 3).Max(k => Math.Max(Math.Abs(glo[k] - rlo[k]), Math.Abs(ghi[k] - rhi[k]))) / size;
        var triangleDifference = Math.Abs(triangles - reference.Triangles) / (double)reference.Triangles;
        var chamfer = MeshCompare.Chamfer(got, reference.Vertices);
        output.WriteLine($"{name} {detail}: {triangles} triangles (reference {reference.Triangles}), box {box:E1}, Chamfer {chamfer:E1}, {job.GetProperty("seconds").GetDouble():F1} s in the service");
        Assert.True(triangleDifference <= 0.01, $"triangles differ {triangleDifference:P2}");
        Assert.True(box <= 0.01, $"the bounding box differs {box:P2}");
        Assert.True(chamfer < 1e-3, $"Chamfer {chamfer:E2}");
        Assert.Equal(0.0, glo[1]);                      // stands on the floor
        Assert.True(triangles <= EngineInfo.MaxTriangles);
    }
}
