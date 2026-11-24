using TripoSR.Service;

ServiceSettings settings;
try { settings = ServiceSettings.FromEnvironment(); }
catch (SettingsException e)
{
    await Console.Error.WriteLineAsync($"error: {e.Message}");
    return 2;
}

using var engine = EngineFactory.Create(settings);
var gate = new FifoGate();
var jobs = new JobStore(engine, gate);
var api = new ServiceApp(settings, jobs, new HealthProbe(settings, gate, engine.Version), gate);
var app = ServiceApp.Build(api, $"http://0.0.0.0:{settings.Port}");
await app.StartAsync();
Console.WriteLine($"triposr-service {settings.Version} listening on :{settings.Port} (weights: {settings.CheckpointDirectory})");
await app.WaitForShutdownAsync();
return 0;
