import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        # Initialisierung der Gewichtsmatrizen basierend auf dem Screenshot
        # self.wih: Gewichte zwischen Input- und Hidden-Layer
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # self.who: Gewichte zwischen Hidden- und Output-Layer
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    def train(self):
        print("I´am training")

    def query(self, inputs_list):
        # Konvertiere die Eingabeliste in ein NumPy-Array
        inputs = np.array(inputs_list)

        # 1. Berechnung der Hidden-Layer Aktivierungen
        # Matrixmultiplikation der Gewichte (wih) mit den Eingaben
        hidden_inputs = np.dot(self.wih, inputs)
        # Anwendung der Sigmoid-Aktivierungsfunktion
        hidden_outputs = special.expit(hidden_inputs)

        # 2. Berechnung der Output-Layer Aktivierungen
        # Matrixmultiplikation der Gewichte (who) mit den Hidden-Outputs
        final_inputs = np.dot(self.who, hidden_outputs)
        # Anwendung der Sigmoid-Aktivierungsfunktion
        final_outputs = special.expit(final_inputs)

        return final_outputs
