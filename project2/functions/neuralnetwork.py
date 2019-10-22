import numpy as np
from functions import *
from HiddenLayer import HiddenLayer
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers = 2
            n_hidden_neurons=100,
            n_categories=2,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        
        self.n_hidden_layers = n_hidden_layers
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        
        self.create_hidden_layers()
        
        self.create_biases_and_weights()
    def create_hidden_layers(self):
        self.hidden_layers = np.zeros((self.n_hidden_layers))
        for i in range(self.n_hidden_layers):
            self.hidden_layers[i] = HiddenLayer(self.n_hidden_neurons)
        
    def create_biases_and_weights(self):
        self.hidden_weights_0 = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias_0 = np.zeros(self.n_hidden_neurons) + 0.01
        
        self.hidden_weights_1 = np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)
        self.hidden_bias_1 = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h_0 = np.matmul(self.X_data, self.hidden_weights_0) + self.hidden_bias_0
        self.a_h_0 = sigmoid(self.z_h_0)
        
        #Should a_h be an attribute to hidden layer object, or should it be globally/locally defined in nn? 
        a_h = self.hidden_layers[0].node_activation(self.X_data)

        for i in range(1,self.n_hidden_layers):
            
            self.hidden_layers[i].node_activation(a_h)
        
        self.z_h_1 = np.matmul(self.a_h_0 ,self.hidden_weights_1) + self.hidden_bias_1
        self.a_h_1 = sigmoid(self.z_h_1)

        self.z_o = np.matmul(self.a_h_1, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def feed_forward_out(self, X):
        # feed-forward for output
        z_h_0 = np.matmul(X, self.hidden_weights_0) + self.hidden_bias_0
        a_h_0 = sigmoid(z_h_0)
        
        z_h_1 = np.matmul(a_h_0, self.hidden_weights_1) + self.hidden_bias_1
        a_h_1 = sigmoid(z_h_1)

        z_o = np.matmul(a_h_1, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        #softmax function
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):

        error_output = self.probabilities - self.Y_data
        
        error_hidden_1 = np.matmul(error_output, self.output_weights.T) * self.a_h_1 * (1 - self.a_h_1)

        error_hidden_0 = np.matmul(error_hidden_1, self.hidden_weights_1.T) * self.a_h_0 * (1 - self.a_h_0)
        

        self.output_weights_gradient = np.matmul(self.a_h_1.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        
        self.hidden_weights_gradient_1 = np.matmul(self.a_h_0.T, error_hidden_1)
        self.hidden_bias_gradient_1 = np.sum(error_hidden_1, axis=0)

        self.hidden_weights_gradient_0 = np.matmul(self.X_data.T, error_hidden_0)
        self.hidden_bias_gradient_0 = np.sum(error_hidden_0, axis=0)
        

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            
            self.hidden_weights_gradient_1 += self.lmbd * self.hidden_weights_1
            
            self.hidden_weights_gradient_0 += self.lmbd * self.hidden_weights_0
            
            
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        
        self.hidden_weights_1 -= self.eta * self.hidden_weights_gradient_1
        self.hidden_bias_1 -= self.eta * self.hidden_bias_gradient_1
        
        self.hidden_weights_0 -= self.eta * self.hidden_weights_gradient_0
        self.hidden_bias_0 -= self.eta * self.hidden_bias_gradient_0

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
