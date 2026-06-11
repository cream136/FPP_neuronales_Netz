# Modulskript: Neuronales Netz

## Verwendete Python-Module

### NumPy
NumPy ist die fundamentale Bibliothek für wissenschaftliches Rechnen in Python. 
- **Wesentliche Inhalte:** Bietet leistungsstarke N-dimensionale Arrays (ndarray) und eine Vielzahl von mathematischen Funktionen zur Manipulation dieser Arrays.
- **Einsatz:** Effiziente Matrixoperationen, welche für die Gewichtungsberechnungen in neuronalen Netzen essenziell sind.

### scipy.special
Ein Teil der SciPy-Bibliothek, der spezialisierte mathematische Funktionen bereitstellt.
- **Wesentliche Inhalte:** Enthält Funktionen wie die Sigmoid-Funktion (oft über `exp` implementiert) oder andere Aktivierungsfunktionen, die in der Datenanalyse und bei neuronalen Netzen benötigt werden.
- **Einsatz:** Implementierung komplexer mathematischer Formeln für die Aktivierung der Neuronen.

### matplotlib.pyplot
Ein Interface zur Erstellung von statischen, animierten und interaktiven Visualisierungen in Python.
- **Wesentliche Inhalte:** Funktionen zum Zeichnen von Graphen, Histogrammen und Streudiagrammen.
- **Einsatz:** Visualisierung des Trainingsfortschritts (z. B. Loss-Kurve) und Analyse der Netzleistung.
