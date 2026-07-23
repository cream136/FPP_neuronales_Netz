#To execute in console before execute this file: 
#1 py -m pip install numpy
#2 py -m pip install spicy
#3 py -m pip install matplotlib


import os
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot

# neural network class definition
class neuralNetwork: 
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.16

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_path = os.path.join(os.path.dirname(__file__), 'mnist_dataset', 'mnist_train.csv')
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Training data file not found: {training_data_path}\nPlease download MNIST CSV and place it in './mnist_dataset/mnist_train.csv'.")

# limit the number of records used for quick testing
train_sample_limit = 10000  # set to None to use the full training set
test_sample_limit = 1000    # set to None to use the full test set

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 7

for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    with open(training_data_path, 'r') as training_data_file:
        for i, record in enumerate(training_data_file):
            if train_sample_limit is not None and i >= train_sample_limit:
                break
            all_values = record.split(',')
            inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        record_count = i + 1
    print(f"  trained on {record_count} records")


# load the mnist test data CSV file into a list
test_data_path = os.path.join(os.path.dirname(__file__), 'mnist_dataset', 'mnist_test.csv')
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}\nPlease download MNIST CSV and place it in './mnist_dataset/mnist_test.csv'.")

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []

with open(test_data_path, 'r') as test_data_file:
    for i, record in enumerate(test_data_file):
        if test_sample_limit is not None and i >= test_sample_limit:
            break
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)
    test_count = i + 1

print(f"Tested on {test_count} records")

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)