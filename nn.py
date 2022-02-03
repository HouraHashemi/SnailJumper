import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.layer_sizes = layer_sizes
        self.layers_parameters = self.initialize_layers(layer_sizes)

    
    def initialize_layers(self, layers):
        layers_params = list()
        for l in range(0,len(layers)-1):
            HLp_HLn_weights = np.random.normal(size=(layers[l+1], layers[l]))
            HLp_HLn_bias = np.zeros((layers[l+1], 1))
            layers_params.append(tuple([HLp_HLn_weights, HLp_HLn_bias]))
        return layers_params  


    def activation(self, inpt):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # Sigmoid
        return 1 / (1 + np.exp(-inpt))


    def forward(self, input_vector):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        a_list = list()
        a = input_vector
        # self.layers_parameters:(w,b)
        for l in range(0,len(self.layers_parameters)):
            w = self.layers_parameters[l][0]
            b = self.layers_parameters[l][1]
            a = self.activation((w @ a + b))
            a_list.append(a)
        return a_list
