import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        # Hier würden normalerweise die Gewichte initialisiert werden
        # self.weights_input_hidden = np.random.randn(self.hidden_nodes, self.input_nodes)
        # self.weights_hidden_output = np.random.randn(self.output_nodes, self.hidden_nodes)

    def train(self):
        print("I´am training")

    def query(self):
        print("I ask…")
