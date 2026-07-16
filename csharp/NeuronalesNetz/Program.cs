using System.Diagnostics;
using NeuronalesNetz;

// Anzahl der Knoten in Eingabe-, verdeckter und Ausgabeschicht
const int InputNodes = 784;
const int HiddenNodes = 100;
const int OutputNodes = 10;

// Lernrate
const double LearningRate = 0.2;

// Wie oft der Trainingsdatensatz durchlaufen wird
const int Epochs = 7;

// Begrenzt die Anzahl der Datensätze für schnelle Testläufe; null = alles nutzen
int? trainSampleLimit = 10000;
int? testSampleLimit = 1000;

var datasetDir = MnistData.FindDatasetDirectory();
if (datasetDir == null)
{
    Console.Error.WriteLine("Ordner 'mnist_dataset' nicht gefunden. Erwartet im Projekt-Wurzelverzeichnis.");
    return 1;
}

var trainingDataPath = Path.Combine(datasetDir, "mnist_train.csv");
var testDataPath = Path.Combine(datasetDir, "mnist_test.csv");

foreach (var path in new[] { trainingDataPath, testDataPath })
{
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"Datei nicht gefunden: {path}");
        Console.Error.WriteLine("Bitte die MNIST-CSV herunterladen und in './mnist_dataset/' ablegen.");
        return 1;
    }
}

var n = new NeuralNetwork(InputNodes, HiddenNodes, OutputNodes, LearningRate);

// Netz trainieren
var stopwatch = Stopwatch.StartNew();
for (int e = 0; e < Epochs; e++)
{
    Console.WriteLine($"Epoche {e + 1}/{Epochs}");

    int recordCount = 0;
    foreach (var record in MnistData.Read(trainingDataPath, trainSampleLimit))
    {
        n.Train(record.Inputs, MnistData.TargetsFor(record.Label, OutputNodes));
        recordCount++;
    }

    Console.WriteLine($"  auf {recordCount} Datensätzen trainiert");
}
Console.WriteLine($"Training dauerte {stopwatch.Elapsed.TotalSeconds:F1} s");

// Netz testen: für jeden Datensatz 1 bei Treffer, 0 bei Fehler
int correct = 0;
int total = 0;
foreach (var record in MnistData.Read(testDataPath, testSampleLimit))
{
    var outputs = n.Query(record.Inputs);
    // Der Index des größten Wertes entspricht dem erkannten Label
    if (outputs.ArgMax() == record.Label) correct++;
    total++;
}

Console.WriteLine($"Auf {total} Datensätzen getestet");
Console.WriteLine($"performance =  {(double)correct / total}");
return 0;
