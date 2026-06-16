# To execute in console before execute this file: 
# 1. python3 -m pip install numpy
# 2. python3 -m pip install scipy
# 3. python3 -m pip install matplotlib

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

    def train(self, inputs_list, targets_list):
        # Wir trainieren über mehrere Epochen, um die Gewichte zu optimieren
        for epoch in range(10000):
            for i in range(len(inputs_list)):
                # --- Forward Pass ---
                inputs = np.array(inputs_list[i])
                target = np.array(targets_list[i])

                # Hidden Layer Aktivierungen
                hidden_inputs = np.dot(self.wih, inputs)
                hidden_outputs = special.expit(hidden_inputs)

                # Output Layer Aktivierungen
                final_inputs = np.dot(self.who, hidden_outputs)
                final_outputs = special.expit(final_inputs)

                # --- Backward Pass (Backpropagation) ---
                
                # 1. Fehler am Output berechnen
                output_errors = target - final_outputs
                
                # Gradient für den Output Layer (Fehler * Ableitung der Sigmoid-Funktion)
                # Ableitung von Sigmoid(x) ist Sigmoid(x) * (1 - Sigmoid(x))
                output_gradient = output_errors * final_outputs * (1.0 - final_outputs)
                
                # Gewichte who aktualisieren: Delta = learning_rate * gradient * hidden_outputs
                self.who += np.outer(output_gradient, hidden_outputs) * self.learning_rate

                # 2. Fehler zurück an Hidden Layer leiten
                hidden_errors = np.dot(self.who.T, output_gradient)
                
                # Gradient für den Hidden Layer
                hidden_gradient = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
                
                # Gewichte wih aktualisieren: Delta = learning_rate * gradient * inputs
                self.wih += np.outer(hidden_gradient, inputs) * self.learning_rate

    def query(self, inputs_list):
        # ...existing code...
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

# create instance of neural network
n = neuralNetwork(2, 3, 1)

# Trainingsdaten für XOR-Problem
inputs_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets_list = [[0], [1], [1], [0]]

# train the neural network
print("Training started... please wait.")
n.train(inputs_list, targets_list)
print("Training finished!")

# query the network to test it
print("\nTesting the network:")
for i in range(len(inputs_list)):
    res = n.query(inputs_list[i])
    print(f"Input: {inputs_list[i]} -> Target: {targets_list[i]} -> Prediction: {res}")
    
print("\nperformance = great")
