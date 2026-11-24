using TripoSR.Service;

namespace TripoSR.Net.Tests;

/// <summary>What the service reads from its environment, and the address it listens on.</summary>
public sealed class ServiceSettingsTests
{
    private static ServiceSettings From(params (string Name, string Value)[] env) =>
        ServiceSettings.From(name => env.Where(e => e.Name == name).Select(e => e.Value).FirstOrDefault());

    [Fact]
    public void ByDefaultItListensOnAllInterfacesLikeTheContainer()
    {
        var s = From();
        Assert.Equal(("0.0.0.0", 8790, "cpu"), (s.Host, s.Port, s.Device));
        Assert.Equal("http://0.0.0.0:8790", s.ListenUrl);
    }

    [Theory]
    [InlineData("127.0.0.1", "http://127.0.0.1:9000")]
    [InlineData("localhost", "http://localhost:9000")]
    [InlineData("::1", "http://[::1]:9000")]
    public void TheHostCanBeLoopbackOnly(string host, string url)
    {
        var s = From(("TRIPOSR_HOST", host), ("TRIPOSR_PORT", "9000"));
        Assert.Equal(host, s.Host);
        Assert.Equal(url, s.ListenUrl);
    }

    [Theory]
    [InlineData("example.com")]
    [InlineData("127.0.0.1:80")]
    [InlineData("http://0.0.0.0")]
    public void AHostThatIsNotAnAddressIsRefusedNamingTheVariable(string host)
    {
        var e = Assert.Throws<SettingsException>(() => From(("TRIPOSR_HOST", host)));
        Assert.Contains("TRIPOSR_HOST", e.Message);
        Assert.Contains(host, e.Message);
    }
}
