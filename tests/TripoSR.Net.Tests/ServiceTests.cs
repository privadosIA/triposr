using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using TripoSR.Service;

namespace TripoSR.Net.Tests;

/// <summary>The service answers the whole contract of the Python container from the first day, with an engine that says it is not ported yet.</summary>
public sealed class ServiceTests : IAsyncLifetime
{
    private static readonly byte[] Png = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0, 0, 0, 0, 0];

    private WebApplication _app = null!;
    private HttpClient _http = null!;
    private string _weights = null!;

    public async Task InitializeAsync() => await Start(token: "");

    private async Task Start(string token, bool withWeights = true)
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
        var engine = EngineFactory.Create(settings);
        var gate = new FifoGate();
        var api = new ServiceApp(settings, new JobStore(engine, gate), new HealthProbe(settings, gate, engine.Version, TimeSpan.Zero), gate);
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
    public async Task WithATokenEveryCallNeedsIt()
    {
        await DisposeAsync();
        await Start("secret");
        Assert.Equal(HttpStatusCode.Unauthorized, (await _http.GetAsync("/health")).StatusCode);
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "secret");
        Assert.Equal(HttpStatusCode.OK, (await _http.GetAsync("/health")).StatusCode);
    }

    [Fact]
    public async Task AJobOfAnEngineThatIsNotPortedFailsWithASentenceAndHasNoModel()
    {
        var r = await _http.PostAsync("/generate?detail=low", new ByteArrayContent(Png));
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var id = (await Json(r)).GetProperty("id").GetString()!;
        JsonElement job = default;
        for (var i = 0; i < 100; i++)
        {
            job = await Json(await _http.GetAsync($"/jobs/{id}"));
            if (job.GetProperty("state").GetString() is "done" or "failed") break;
            await Task.Delay(20);
        }
        Assert.Equal("failed", job.GetProperty("state").GetString());
        Assert.Contains("not ported", job.GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.Conflict, (await _http.GetAsync($"/jobs/{id}/model")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await _http.DeleteAsync($"/jobs/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync($"/jobs/{id}")).StatusCode);
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
