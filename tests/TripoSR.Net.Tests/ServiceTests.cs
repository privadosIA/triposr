using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using TripoSR.Mesh;
using TripoSR.Service;

namespace TripoSR.Net.Tests;

/// <summary>The service answers the whole contract of the Python container, here with a stand-in engine (the real one is in <c>ServiceRealEngineTests</c>).</summary>
public sealed class ServiceTests : IAsyncLifetime
{
    private static readonly byte[] Png = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0, 0, 0, 0, 0];

    private WebApplication _app = null!;
    private HttpClient _http = null!;
    private string _weights = null!;

    public async Task InitializeAsync() => await Start(token: "");

    private async Task Start(string token, bool withWeights = true, IModelEngine? engine = null)
    {
        _weights = Path.Combine(Path.GetTempPath(), "triposr-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_weights);
        if (withWeights)
            foreach (var f in EngineInfo.RequiredFiles)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(_weights, f))!);
                File.WriteAllBytes(Path.Combine(_weights, f), []);
            }
        var settings = new ServiceSettings(_weights, token, 0, "test", "cpu");
        engine ??= new FakeEngine((_, detail) => new GenerationResult(
            GlbWriter.Write(new ColoredMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2], new byte[12])), detail, EngineInfo.DetailSizes[detail], 1));
        var gate = new FifoGate();
        var probe = new HealthProbe(settings, gate, engine.Version, TimeSpan.Zero);
        var api = new ServiceApp(settings, new JobStore(engine, gate, probe.MissingFiles), probe, gate);
        _app = ServiceApp.Build(api, "http://127.0.0.1:0");
        await _app.StartAsync();
        var address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address) };
    }

    public async Task DisposeAsync()
    {
        _http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        Directory.Delete(_weights, true);
    }

    private async Task<JsonElement> Json(HttpResponseMessage r) => JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;

    [Fact]
    public async Task HealthSaysItIsReadyWhenTheWeightsAreThere()
    {
        var r = await _http.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        var h = await Json(r);
        Assert.True(h.GetProperty("ready").GetBoolean());
        Assert.Equal("test", h.GetProperty("version").GetString());
        Assert.Equal(0, h.GetProperty("missing").GetArrayLength());
        Assert.True(h.GetProperty("engines").GetProperty(EngineInfo.Id).GetProperty("ready").GetBoolean());
    }

    [Fact]
    public async Task EnginesNameTheVersionThatDecidesAModel()
    {
        var e = (await Json(await _http.GetAsync("/engines"))).GetProperty("engines").GetProperty(EngineInfo.Id);
        Assert.True(e.GetProperty("built").GetBoolean());
        Assert.Equal(EngineInfo.Version, e.GetProperty("version").GetString());
        Assert.StartsWith(EngineInfo.Id + "-net@", EngineInfo.Version);
    }

    [Fact]
    public async Task HealthNamesTheMissingFiles()
    {
        await DisposeAsync();
        await Start("", withWeights: false);
        var h = await Json(await _http.GetAsync("/health"));
        Assert.False(h.GetProperty("ready").GetBoolean());
        Assert.Equal(EngineInfo.RequiredFiles.Count, h.GetProperty("missing").GetArrayLength());
    }

    [Fact]
    public async Task AJobIsRefusedWhenTheWeightsAreNotComplete()
    {
        await DisposeAsync();
        await Start("", withWeights: false);
        var r = await _http.PostAsync("/generate", new ByteArrayContent(Png));
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        var said = (await Json(r)).GetProperty("error").GetString();
        Assert.Contains("the weights of triposr are not complete", said);
        Assert.Contains("triposr/model.ckpt", said);
    }

    [Fact]
    public async Task WithATokenEveryCallNeedsIt()
    {
        await DisposeAsync();
        await Start("secret");
        Assert.Equal(HttpStatusCode.Unauthorized, (await _http.GetAsync("/health")).StatusCode);
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "secret");
        Assert.Equal(HttpStatusCode.OK, (await _http.GetAsync("/health")).StatusCode);
    }

    private async Task<JsonElement> WaitForJob(string id)
    {
        JsonElement job = default;
        for (var i = 0; i < 200; i++)
        {
            job = await Json(await _http.GetAsync($"/jobs/{id}"));
            if (job.GetProperty("state").GetString() is "done" or "failed") break;
            await Task.Delay(20);
        }
        return job;
    }

    [Fact]
    public async Task APictureBecomesAJobThatEndsDoneWithItsModel()
    {
        var r = await _http.PostAsync("/generate?detail=low", new ByteArrayContent(Png));
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var id = (await Json(r)).GetProperty("id").GetString()!;
        var job = await WaitForJob(id);
        Assert.Equal("done", job.GetProperty("state").GetString());
        Assert.Equal(("low", "low", 128, 1, EngineInfo.Version), (job.GetProperty("detail").GetString(), job.GetProperty("detail_used").GetString(), job.GetProperty("resolution").GetInt32(), job.GetProperty("triangles").GetInt32(), job.GetProperty("engine_version").GetString()));
        var model = await _http.GetAsync($"/jobs/{id}/model");
        Assert.Equal("model/gltf-binary", model.Content.Headers.ContentType?.MediaType);
        Assert.Equal("glTF", System.Text.Encoding.ASCII.GetString((await model.Content.ReadAsByteArrayAsync())[..4]));
        Assert.Equal(HttpStatusCode.OK, (await _http.DeleteAsync($"/jobs/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync($"/jobs/{id}")).StatusCode);
    }

    [Fact]
    public async Task AnEngineThatRefusesFailsTheJobWithASentenceAndLeavesNoModel()
    {
        await DisposeAsync();
        await Start("", engine: new FakeEngine((_, _) => throw new RefusedException("the engine made no surface: the picture may not show an object on its own")));
        var id = (await Json(await _http.PostAsync("/generate?detail=low", new ByteArrayContent(Png)))).GetProperty("id").GetString()!;
        var job = await WaitForJob(id);
        Assert.Equal("failed", job.GetProperty("state").GetString());
        Assert.Contains("no surface", job.GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.Conflict, (await _http.GetAsync($"/jobs/{id}/model")).StatusCode);
    }

    [Fact]
    public async Task AnUnexpectedFailureIsASentenceToo()
    {
        await DisposeAsync();
        await Start("", engine: new FakeEngine((_, _) => throw new OutOfMemoryException("CUDA out of memory")));
        var id = (await Json(await _http.PostAsync("/generate", new ByteArrayContent(Png)))).GetProperty("id").GetString()!;
        var job = await WaitForJob(id);
        Assert.Equal("failed", job.GetProperty("state").GetString());
        Assert.Equal("OutOfMemoryException: CUDA out of memory", job.GetProperty("error").GetString());
    }

    [Theory]
    [InlineData("/generate?engine=other", "this service makes")]
    [InlineData("/generate?detail=ultra", "detail must be")]
    public async Task ARequestThatCannotBeMadeIs400(string url, string said)
    {
        var r = await _http.PostAsync(url, new ByteArrayContent(Png));
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Contains(said, (await Json(r)).GetProperty("error").GetString());
    }

    [Fact]
    public async Task ABodyThatIsNotAPictureIs400AndAnEmptyOneToo()
    {
        Assert.Equal(HttpStatusCode.BadRequest, (await _http.PostAsync("/generate", new ByteArrayContent("not a picture at all"u8.ToArray()))).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await _http.PostAsync("/generate", new ByteArrayContent([]))).StatusCode);
    }

    [Fact]
    public async Task UnknownThingsAre404()
    {
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync("/jobs/none")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync("/nowhere")).StatusCode);
    }

    [Fact]
    public async Task TheDescriptionIsServedAndListsTheSameCallsAsTheService()
    {
        var yaml = await (await _http.GetAsync("/openapi.yaml")).Content.ReadAsStringAsync();
        foreach (var route in ServiceApp.RouteTable)
        {
            var (method, path) = (route.Split(' ')[0].ToLowerInvariant(), route.Split(' ')[1]);
            var at = yaml.IndexOf($"\n  {path}:", StringComparison.Ordinal);
            Assert.True(at >= 0, $"{path} is not in openapi.yaml");
            var next = yaml.IndexOf("\n  /", at + 1, StringComparison.Ordinal);
            var block = next < 0 ? yaml[at..] : yaml[at..next];
            Assert.Contains($"\n    {method}:", block);
        }
        Assert.Equal(HttpStatusCode.OK, (await _http.GetAsync("/docs")).StatusCode);
    }

    [Fact]
    public void TheDetailsAreTheThreeOfTheContract() =>
        Assert.Equal(new[] { "high", "low", "normal" }, EngineInfo.DetailSizes.Keys.Order().ToArray());
}

/// <summary>An engine for the tests of the HTTP side: it makes whatever the function says.</summary>
internal sealed class FakeEngine(Func<byte[], string, GenerationResult> make) : IModelEngine
{
    public string Version => EngineInfo.Version;
    public GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel) => make(picture, detail);
    public void Dispose() { }
}
