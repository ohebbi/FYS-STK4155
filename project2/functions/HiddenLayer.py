
from functions.functions import *
import numpy as np

class HiddenLayer:
    def __init__(self,
                 weights,
                 bias,
                 n_neurons):
        self.weights = weights
        self.bias = bias
        self.n_neurons = n_neurons
        
    def node_activation(self,X_data):
        self.z_h_0 = np.matmul(X_data, self.weights) + self.bias
        self.a_h_0 = sigmoid(self.z_h_0)
        return a_h_0 