using System.Globalization;

namespace NeuronalesNetz;

/// <summary>Ein Datensatz aus der MNIST-CSV: das Label und die 784 Pixelwerte.</summary>
public readonly record struct MnistRecord(int Label, double[] Inputs);

public static class MnistData
{
    /// <summary>
    /// Liest die CSV zeilenweise (nicht komplett in den Speicher) und skaliert die
    /// Pixel von 0..255 auf 0.01..1.0 — wie im Python-Original.
    /// </summary>
    public static IEnumerable<MnistRecord> Read(string path, int? sampleLimit = null)
    {
        int count = 0;
        foreach (var line in File.ReadLines(path))
        {
            if (sampleLimit.HasValue && count >= sampleLimit.Value) yield break;
            if (string.IsNullOrWhiteSpace(line)) continue;

            var allValues = line.Split(',');
            int label = int.Parse(allValues[0], CultureInfo.InvariantCulture);

            var inputs = new double[allValues.Length - 1];
            for (int i = 0; i < inputs.Length; i++)
            {
                double pixel = double.Parse(allValues[i + 1], CultureInfo.InvariantCulture);
                inputs[i] = (pixel / 255.0 * 0.99) + 0.01;
            }

            count++;
            yield return new MnistRecord(label, inputs);
        }
    }

    /// <summary>
    /// Zielvektor: überall 0.01, nur am Index des Labels 0.99. Die Extremwerte 0 und 1
    /// erreicht das Sigmoid nie, deshalb wird nicht auf sie trainiert.
    /// </summary>
    public static double[] TargetsFor(int label, int outputNodes)
    {
        var targets = new double[outputNodes];
        Array.Fill(targets, 0.01);
        targets[label] = 0.99;
        return targets;
    }

    /// <summary>
    /// Sucht den mnist_dataset-Ordner ausgehend vom Programmverzeichnis nach oben.
    /// Die Binary liegt unter bin/Debug/netX.0, das Dataset im Projekt-Wurzelordner.
    /// </summary>
    public static string? FindDatasetDirectory()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "mnist_dataset");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }
}
