using System.Globalization;

namespace TripoSR.Service;

/// <summary>A setting that cannot be used. The message names the environment variable.</summary>
internal sealed class SettingsException(string message) : Exception(message);

/// <summary>What the service is told by its environment, with the variable names of the Python container (<c>MODEL3D_*</c> there, <c>TRIPOSR_*</c> here).</summary>
/// <param name="CheckpointDirectory">TRIPOSR_CKPT: where the weights are (the layout of the container's <c>/ckpt</c>).</param>
/// <param name="Token">TRIPOSR_TOKEN: bearer token; empty leaves the API open.</param>
/// <param name="Port">TRIPOSR_PORT: the port to listen on, on all interfaces.</param>
/// <param name="Version">TRIPOSR_VERSION: the version string health reports (the version of a model is <see cref="TripoSR.EngineInfo.Version"/>).</param>
/// <param name="Device">TRIPOSR_DEVICE: cpu, cuda or cuda:N.</param>
/// <param name="Host">TRIPOSR_HOST: the address to listen on: an IP address (<c>0.0.0.0</c>, all interfaces, as the Python container does, by default; <c>127.0.0.1</c> for this machine only) or <c>localhost</c>.</param>
internal sealed record ServiceSettings(string CheckpointDirectory, string Token, int Port, string Version, string Device, string Host = "0.0.0.0")
{
    public const string CkptVariable = "TRIPOSR_CKPT", TokenVariable = "TRIPOSR_TOKEN", PortVariable = "TRIPOSR_PORT", VersionVariable = "TRIPOSR_VERSION", DeviceVariable = "TRIPOSR_DEVICE", HostVariable = "TRIPOSR_HOST";

    /// <summary>The address as it goes in a URL (an IPv6 address in brackets).</summary>
    public string UrlHost => Host.Contains(':') ? $"[{Host}]" : Host;

    /// <summary>The URL the service listens on.</summary>
    public string ListenUrl => $"http://{UrlHost}:{Port}";

    public static ServiceSettings FromEnvironment() => From(Environment.GetEnvironmentVariable);

    public static ServiceSettings From(Func<string, string?> get)
    {
        string Text(string name, string fallback) => get(name) is { Length: > 0 } v ? v : fallback;

        var port = 8790;
        if (get(PortVariable) is { Length: > 0 } p
            && (!int.TryParse(p, NumberStyles.None, CultureInfo.InvariantCulture, out port) || port < 1 || port > 65535))
            throw new SettingsException($"{PortVariable} must be a port number from 1 to 65535, got '{p}'");

        var device = Text(DeviceVariable, "cpu");
        if (!IsDevice(device)) throw new SettingsException($"{DeviceVariable} must be cpu, cuda or cuda:N, got '{device}'");

        var host = Text(HostVariable, "0.0.0.0");
        if (!IsHost(host)) throw new SettingsException($"{HostVariable} must be an IP address (0.0.0.0 for all interfaces, 127.0.0.1 for this machine only) or localhost, got '{host}'");

        return new ServiceSettings(Text(CkptVariable, "/ckpt"), get(TokenVariable) ?? "", port, Text(VersionVariable, "1"), device.ToLowerInvariant(), host);
    }

    public static bool IsHost(string h) => h.Equals("localhost", StringComparison.OrdinalIgnoreCase) || System.Net.IPAddress.TryParse(h, out _);

    public static bool IsDevice(string d) =>
        d.Equals("cpu", StringComparison.OrdinalIgnoreCase) || d.Equals("cuda", StringComparison.OrdinalIgnoreCase)
        || (d.StartsWith("cuda:", StringComparison.OrdinalIgnoreCase) && int.TryParse(d.AsSpan(5), NumberStyles.None, CultureInfo.InvariantCulture, out _));
}
