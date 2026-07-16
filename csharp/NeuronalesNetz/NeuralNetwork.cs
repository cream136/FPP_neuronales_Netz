namespace NeuronalesNetz;

/// <summary>
/// Dreischichtiges neuronales Netz (Eingabe, eine verdeckte Schicht, Ausgabe)
/// mit Sigmoid-Aktivierung und Backpropagation.
/// Portierung der Python-Klasse neuralNetwork aus neuronalesNetzwerkSchuleDoku.py.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly int _inputNodes;
    private readonly int _hiddenNodes;
    private readonly int _outputNodes;
    private readonly double _learningRate;

    // Gewichtsmatrizen: wih = input -> hidden, who = hidden -> output.
    // Gewicht w_i_j verbindet Knoten i mit Knoten j der nächsten Schicht.
    private readonly Matrix _wih;
    private readonly Matrix _who;

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate, int? seed = null)
    {
        _inputNodes = inputNodes;
        _hiddenNodes = hiddenNodes;
        _outputNodes = outputNodes;
        _learningRate = learningRate;

        var random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Startgewichte normalverteilt um 0, Streuung 1/sqrt(Anzahl eingehender Verbindungen)
        _wih = Matrix.RandomNormal(random, 0.0, Math.Pow(_inputNodes, -0.5), _hiddenNodes, _inputNodes);
        _who = Matrix.RandomNormal(random, 0.0, Math.Pow(_hiddenNodes, -0.5), _outputNodes, _hiddenNodes);
    }

    public int InputNodes => _inputNodes;
    public int OutputNodes => _outputNodes;

    /// <summary>Sigmoid — entspricht scipy.special.expit.</summary>
    private static double ActivationFunction(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>Ein Trainingsschritt mit einem Datensatz.</summary>
    public void Train(double[] inputsList, double[] targetsList)
    {
        if (inputsList.Length != _inputNodes)
        {
            throw new ArgumentException($"Erwartet {_inputNodes} Eingabewerte, bekommen {inputsList.Length}.", nameof(inputsList));
        }
        if (targetsList.Length != _outputNodes)
        {
            throw new ArgumentException($"Erwartet {_outputNodes} Zielwerte, bekommen {targetsList.Length}.", nameof(targetsList));
        }

        var inputs = Matrix.ColumnVector(inputsList);
        var targets = Matrix.ColumnVector(targetsList);

        // Vorwärtsdurchlauf
        var hiddenOutputs = Matrix.Dot(_wih, inputs).Map(ActivationFunction);
        var finalOutputs = Matrix.Dot(_who, hiddenOutputs).Map(ActivationFunction);

        // Fehler der Ausgabeschicht ist (Ziel - Ist)
        var outputErrors = Matrix.Subtract(targets, finalOutputs);
        // Fehler der verdeckten Schicht: Ausgabefehler, nach Gewichten aufgeteilt
        var hiddenErrors = Matrix.TransposedDot(_who, outputErrors);

        // Gewichte hidden -> output aktualisieren
        _who.AddScaled(
            Matrix.DotTransposed(Matrix.SigmoidGradient(outputErrors, finalOutputs), hiddenOutputs),
            _learningRate);

        // Gewichte input -> hidden aktualisieren
        _wih.AddScaled(
            Matrix.DotTransposed(Matrix.SigmoidGradient(hiddenErrors, hiddenOutputs), inputs),
            _learningRate);
    }

    /// <summary>Fragt das Netz ab und liefert die Aktivierung der Ausgabeschicht.</summary>
    public Matrix Query(double[] inputsList)
    {
        if (inputsList.Length != _inputNodes)
        {
            throw new ArgumentException($"Erwartet {_inputNodes} Eingabewerte, bekommen {inputsList.Length}.", nameof(inputsList));
        }

        var inputs = Matrix.ColumnVector(inputsList);
        var hiddenOutputs = Matrix.Dot(_wih, inputs).Map(ActivationFunction);
        return Matrix.Dot(_who, hiddenOutputs).Map(ActivationFunction);
    }
}
