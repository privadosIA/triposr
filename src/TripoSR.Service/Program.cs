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
var health = new HealthProbe(settings, gate, engine.Version);
var jobs = new JobStore(engine, gate, health.MissingFiles);
var api = new ServiceApp(settings, jobs, health, gate);
var app = ServiceApp.Build(api, settings.ListenUrl);
await app.StartAsync();
Console.WriteLine($"triposr-service {settings.Version} listening on {settings.Host}:{settings.Port} (weights: {settings.CheckpointDirectory})");
await app.WaitForShutdownAsync();
return 0;
