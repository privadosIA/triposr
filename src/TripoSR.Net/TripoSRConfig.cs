namespace TripoSR;

/// <summary>The numbers of <c>triposr/config.yaml</c> (the model is one fixed network; a test checks these against the file).</summary>
public sealed record TripoSRConfig
{
    public int CondImageSize { get; init; } = 512;

    // tokenizer
    public int PlaneSize { get; init; } = 32;
    public int TokenChannels { get; init; } = 1024;

    // backbone
    public int Heads { get; init; } = 16;
    public int HeadDim { get; init; } = 64;
    public int Layers { get; init; } = 16;
    public int CrossAttentionDim { get; init; } = 768;
    public int NormGroups { get; init; } = 32;

    // post processor
    public int SceneChannels { get; init; } = 40;

    // decoder (the field)
    public int DecoderHidden { get; init; } = 64;
    public int DecoderHiddenLayers { get; init; } = 9;

    // renderer
    public double Radius { get; init; } = 0.87;
    public double DensityBias { get; init; } = -1.0;

    public int InnerDim => Heads * HeadDim;

    /// <summary>
    /// Checks the text of <c>triposr/config.yaml</c> against the network this port is: it is one fixed network, so a file that describes another one is refused naming what differs.
    /// </summary>
    /// <exception cref="InvalidDataException">The file does not describe this network.</exception>
    public void Verify(string yaml, string source = "config.yaml")
    {
        var problems = new List<string>();
        void Number(string key, double expected)
        {
            var m = System.Text.RegularExpressions.Regex.Match(yaml, @"^\s*" + key + @":\s*(-?[0-9.]+)", System.Text.RegularExpressions.RegexOptions.Multiline);
            if (!m.Success) problems.Add($"{key} is not in the file");
            else if (Math.Abs(double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) - expected) > 1e-9) problems.Add($"{key} is {m.Groups[1].Value}, this port is {expected}");
        }
        Number("cond_image_size", CondImageSize);
        Number("plane_size", PlaneSize);
        Number("num_channels", TokenChannels);
        Number("num_attention_heads", Heads);
        Number("attention_head_dim", HeadDim);
        Number("num_layers", Layers);
        Number("cross_attention_dim", CrossAttentionDim);
        Number("out_channels", SceneChannels);
        Number("n_neurons", DecoderHidden);
        Number("n_hidden_layers", DecoderHiddenLayers);
        Number("radius", Radius);
        Number("density_bias", DensityBias);
        foreach (var (key, value) in new[] { ("density_activation", "exp"), ("color_activation", null), ("feature_reduction", "concat"), ("activation", "silu") })
        {
            if (value is null) continue;
            if (!System.Text.RegularExpressions.Regex.IsMatch(yaml, @"^\s*" + key + @":\s*" + value + @"\b", System.Text.RegularExpressions.RegexOptions.Multiline)) problems.Add($"{key} is not {value}");
        }
        if (problems.Count > 0) throw new InvalidDataException($"{source} does not describe the TripoSR network of this port: {string.Join("; ", problems)}");
    }
}
