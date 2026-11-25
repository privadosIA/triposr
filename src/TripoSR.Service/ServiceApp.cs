using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Server.Kestrel.Core;

namespace TripoSR.Service;

/// <summary>The HTTP interface of the service: the calls of the Python container <c>docker/model3d</c> of Veritmo, with its paths, bodies and status codes.</summary>
internal sealed class ServiceApp
{
    /// <summary>The calls this class answers (keep in step with <see cref="Handle"/>; a test compares it with the OpenAPI description).</summary>
    public static readonly string[] RouteTable =
        { "GET /health", "GET /engines", "POST /generate", "GET /jobs/{id}", "GET /jobs/{id}/model", "DELETE /jobs/{id}" };

    private readonly ServiceSettings _settings;
    private readonly JobStore _jobs;
    private readonly HealthProbe _health;
    private readonly FifoGate _gate;
    private readonly byte[] _expectedAuth;

    public ServiceApp(ServiceSettings settings, JobStore jobs, HealthProbe health, FifoGate gate)
    {
        _settings = settings;
        _jobs = jobs;
        _health = health;
        _gate = gate;
        _expectedAuth = Encoding.UTF8.GetBytes("Bearer " + settings.Token);
    }

    /// <summary>Builds the web application listening on <paramref name="url"/>.</summary>
    public static WebApplication Build(ServiceApp api, string url)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls(url);
        builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = null); // the body is counted here against the limit of the picture
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();
        app.Run(api.Handle);
        return app;
    }

    private async Task Handle(HttpContext ctx)
    {
        try
        {
            var path = ctx.Request.Path.Value ?? "/";
            if (path.Length > 1) path = path.TrimEnd('/');
            var method = ctx.Request.Method;

            if (await Docs.TryServe(ctx, path, method)) return;

            if (_settings.Token.Length > 0 && !Authorized(ctx)) { await Json(ctx, 401, new ErrorBody("a token is needed")); return; }

            if (method == "GET" && path == "/health") { await Json(ctx, 200, _health.Get()); return; }
            if (method == "GET" && path == "/engines") { await Json(ctx, 200, new EnginesBody(_health.Engines())); return; }
            if (method == "POST" && path == "/generate") { await Generate(ctx); return; }
            var parts = path.Trim('/').Split('/');
            if (parts.Length >= 2 && parts[0] == "jobs")
            {
                var job = _jobs.Get(parts[1]);
                if (job is null) { await Json(ctx, 404, new ErrorBody("no such job")); return; }
                if (method == "GET" && parts.Length == 2) { await Json(ctx, 200, job.View(DateTime.UtcNow)); return; }
                if (method == "GET" && parts.Length == 3 && parts[2] == "model") { await Model(ctx, job); return; }
                if (method == "DELETE" && parts.Length == 2) { _jobs.Forget(job.Id); await Json(ctx, 200, new DeletedBody(true)); return; }
            }
            await Json(ctx, 404, new ErrorBody("no such call"));
        }
        catch (Exception e) when (!ctx.Response.HasStarted)
        {
            await Json(ctx, 500, new ErrorBody(e.Message));
        }
    }

    private bool Authorized(HttpContext ctx)
    {
        var given = Encoding.UTF8.GetBytes(ctx.Request.Headers.Authorization.ToString());
        return given.Length == _expectedAuth.Length && CryptographicOperations.FixedTimeEquals(given, _expectedAuth);
    }

    private async Task Generate(HttpContext ctx)
    {
        var query = ctx.Request.Query;
        var engine = query["engine"].FirstOrDefault() ?? EngineInfo.Id;
        var detail = query["detail"].FirstOrDefault() ?? "normal";
        if (ctx.Request.ContentLength > EngineInfo.MaxPictureBytes) { await Json(ctx, 413, new ErrorBody($"the picture is larger than {EngineInfo.MaxPictureBytes >> 20} MB")); return; }
        var buffer = new MemoryStream();
        var chunk = new byte[64 * 1024];
        int n;
        while ((n = await ctx.Request.Body.ReadAsync(chunk, ctx.RequestAborted)) > 0)
        {
            await buffer.WriteAsync(chunk.AsMemory(0, n), ctx.RequestAborted);
            if (buffer.Length > EngineInfo.MaxPictureBytes) { await Json(ctx, 413, new ErrorBody($"the picture is larger than {EngineInfo.MaxPictureBytes >> 20} MB")); return; }
        }
        Job job;
        try { job = _jobs.Submit(engine, detail, buffer.ToArray()); }
        catch (RefusedException e) { await Json(ctx, 400, new ErrorBody(e.Message)); return; }
        await Json(ctx, 202, new SubmittedBody(job.Id));
    }

    private static async Task Model(HttpContext ctx, Job job)
    {
        if (job.State != "done" || job.Glb is null) { await Json(ctx, 409, new ErrorBody($"the model is not ready (state: {job.State})")); return; }
        ctx.Response.StatusCode = 200;
        ctx.Response.ContentType = "model/gltf-binary";
        ctx.Response.ContentLength = job.Glb.Length;
        await ctx.Response.Body.WriteAsync(job.Glb, ctx.RequestAborted);
    }

    // ---- JSON ----------------------------------------------------------------------------------------------------------------------

    private static readonly JsonSerializerOptions JsonOptions = new() { DefaultIgnoreCondition = JsonIgnoreCondition.Never };

    internal static async Task Json<T>(HttpContext ctx, int status, T body)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(body, JsonOptions);
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        ctx.Response.ContentLength = bytes.Length;
        await ctx.Response.Body.WriteAsync(bytes, ctx.RequestAborted);
    }

    internal sealed record ErrorBody([property: JsonPropertyName("error")] string Error);

    internal sealed record EnginesBody([property: JsonPropertyName("engines")] IReadOnlyDictionary<string, EngineView> Engines);

    internal sealed record SubmittedBody([property: JsonPropertyName("id")] string Id);

    internal sealed record DeletedBody([property: JsonPropertyName("deleted")] bool Deleted);
}

/// <summary>The documentation endpoints (<c>/docs</c>, <c>/openapi.yaml</c>), served from the executable itself, without a token.</summary>
internal static class Docs
{
    private static readonly Assembly Assembly = typeof(Docs).Assembly;

    private static string ContentType(string name) => Path.GetExtension(name) switch
    {
        ".html" => "text/html; charset=utf-8",
        ".js" => "text/javascript; charset=utf-8",
        ".css" => "text/css; charset=utf-8",
        ".png" => "image/png",
        ".yaml" => "application/yaml; charset=utf-8",
        _ => "text/plain; charset=utf-8",
    };

    /// <summary>The text of the OpenAPI description.</summary>
    public static string OpenApi
    {
        get
        {
            using var s = Assembly.GetManifestResourceStream("openapi.yaml")!;
            using var r = new StreamReader(s, Encoding.UTF8);
            return r.ReadToEnd();
        }
    }

    /// <summary>Serves the embedded documentation or API description when <paramref name="path"/> is one of them; false when it is not (or the method is not GET/HEAD).</summary>
    public static async Task<bool> TryServe(HttpContext ctx, string path, string method)
    {
        string? resource = path switch
        {
            "/openapi.yaml" => "openapi.yaml",
            "/docs" => "docs/index.html",
            _ when path.StartsWith("/docs/", StringComparison.Ordinal) && !path.Contains("..") => "docs/" + path["/docs/".Length..],
            _ => null,
        };
        if (resource is null || (method != "GET" && method != "HEAD")) return false;
        var stream = Assembly.GetManifestResourceStream(resource);
        if (stream is null) return false;
        await Send(ctx, stream, resource);
        return true;
    }

    private static async Task Send(HttpContext ctx, Stream stream, string resource)
    {
        await using (stream)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = ContentType(resource);
            ctx.Response.ContentLength = stream.Length;
            if (ctx.Request.Method != "HEAD") await stream.CopyToAsync(ctx.Response.Body, ctx.RequestAborted);
        }
    }
}
