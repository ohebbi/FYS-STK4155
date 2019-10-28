
from functions.functions import *
import numpy as np

class HiddenLayer:
    def __init__(self,
                 n_input,
                 n_output
                 ):

        self.n_input = n_input
        self.n_output = n_output
        #self.a_h = 0
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.weights = np.random.randn(self.n_input, self.n_output)

        self.bias = np.zeros(self.n_output) + 0.01


    def node_activation(self,X_data):
        self.z_h = np.matmul(X_data, self.weights) + self.bias
        self.a_h = sigmoid(self.z_h)
        return self.a_h

    def node_activation_out(self,X_data):
        z_h = np.matmul(X_data, self.weights) + self.bias
        a_h = sigmoid(self.z_h)
        return a_h

    def error_layer(self, error_output, weights):
        self.error_hidden = np.matmul(error_output, weights.T) * self.a_h * (1 - self.a_h)
        return self.error_hidden

    def gradients(self, a_h, lmbd):
        self.weights_gradient = np.matmul(a_h.T, self.error_hidden)
        self.bias_gradient = np.sum(self.error_hidden, axis=0)

        if lmbd > 0.0:
            self.weights_gradient += lmbd * self.weights

    def update_weights(self, eta):
        self.weights -= eta * self.weights_gradient
        self.bias -= eta * self.bias_gradient
