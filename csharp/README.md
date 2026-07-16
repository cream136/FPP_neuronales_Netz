# Neuronales Netz — C#-Portierung

Portierung von `neuronalesNetzwerkSchuleDoku.py` nach C#. Dreischichtiges Netz
(784 → 100 → 10) mit Sigmoid-Aktivierung und Backpropagation, trainiert auf MNIST.

Keine externen Abhängigkeiten: die benötigten NumPy-Operationen sind in `Matrix.cs`
selbst implementiert.

## Ausführen

```
cd NeuronalesNetz
dotnet run -c Release
```

Das Dataset wird automatisch gesucht — der Ordner `mnist_dataset/` im Projekt-Wurzel-
verzeichnis wird gefunden, ohne dass ein Pfad angepasst werden muss.

Ein Lauf mit den Standardeinstellungen (10.000 Trainings-, 1.000 Testdatensätze,
7 Epochen) dauert etwa 50 Sekunden und erreicht rund 94,5 % Erkennungsrate —
gleichauf mit der Python-Version (94,6 %).

## Dateien

| Datei | Inhalt |
| --- | --- |
| `Matrix.cs` | Matrixklasse mit den Rechenoperationen, die NumPy sonst liefert |
| `NeuralNetwork.cs` | Das Netz selbst: `Train` und `Query`, entspricht der Python-Klasse `neuralNetwork` |
| `MnistData.cs` | CSV einlesen, Pixel skalieren, Zielvektoren bauen, Dataset finden |
| `Program.cs` | Ablauf: Netz anlegen, trainieren, testen — entspricht dem Skriptteil unten in der .py |

## Parameter

Alle oben in `Program.cs`:

```csharp
const int HiddenNodes = 100;
const double LearningRate = 0.2;
const int Epochs = 7;
int? trainSampleLimit = 10000;  // null = kompletter Trainingssatz
int? testSampleLimit = 1000;    // null = kompletter Testsatz
```

Für einen Lauf auf dem vollen Datensatz beide Limits auf `null` setzen.

## Unterschiede zum Python-Original

- **Gewichtsinitialisierung**: NumPy nutzt den Mersenne-Twister, .NET einen anderen
  Generator. Die Verteilung ist dieselbe (Normalverteilung, Streuung 1/√n, per
  Box-Muller erzeugt), die konkreten Startgewichte sind es nicht. Für einen
  reproduzierbaren Lauf nimmt der `NeuralNetwork`-Konstruktor optional einen `seed`.
- **Sigmoid**: `1 / (1 + e^-x)` direkt statt `scipy.special.expit` — mathematisch
  identisch.
- **Datenhaltung**: Die CSV wird zeilenweise gelesen (`File.ReadLines`) statt komplett
  in eine Liste. Spart bei der 100-MB-Trainingsdatei einiges an Speicher.
- **Validierung**: `Train` und `Query` prüfen die Länge der übergebenen Vektoren gegen
  die konfigurierten Knotenzahlen und werfen sonst eine aussagekräftige Exception.

## Anmerkungen zum ursprünglichen Python-Code

Beim Portieren aufgefallen, in der C#-Fassung jeweils behoben:

- **`neuralNetwork.py`** trainiert nicht wirklich: die `train`-Methode berechnet den
  Vorwärtsdurchlauf und gibt ihn aus, aktualisiert aber keine Gewichte. Ausserdem
  übergibt der Aufruf am Dateiende zwei Datensätze auf einmal an ein Netz, dessen
  `train` nur einen erwartet. Sieht nach einem Zwischenstand aus.
- **`KNN_mnist_data.py`** hat hartkodierte Pfade auf ein fremdes Windows-Profil
  (`C:/Users/Zobel/OneDrive - .../mnist_test.csv`) und läuft deshalb hier nicht.
- **`neuronalesNetzwerkSchuleDoku.py`** zählt beim Abbruch am Sample-Limit einen
  Datensatz zu viel (`record_count = i + 1`, wobei `i` bereits der Index der nicht mehr
  trainierten Zeile ist). Deshalb meldet die Python-Version "10001 records", obwohl sie
  auf 10000 trainiert hat — reiner Anzeigefehler, das Ergebnis stimmt.
- Alle drei Dateien importieren `matplotlib.pyplot`, ohne es zu benutzen.
