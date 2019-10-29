import numpy as np
from functions import *
from functions.hiddenlayer import HiddenLayer
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers = 2,
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
        self.hidden_layers = np.empty(self.n_hidden_layers+1, dtype = "object")

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_hidden_layers()

    def create_hidden_layers(self):
        if self.n_hidden_layers <= 0:
            msg = 'Number of hidden layers has to 1 or more. n_hidden_layers = ', self.n_hidden_layers
            raise ValueError(msg)
        self.hidden_layers[0] = HiddenLayer(
                                            n_input = self.n_features,
                                            n_output = self.n_hidden_neurons
                                            )

        for i in range(1,self.n_hidden_layers):
            self.hidden_layers[i] = HiddenLayer(
                                                n_input = self.n_hidden_neurons,
                                                n_output = self.n_hidden_neurons
                                                )

        self.hidden_layers[-1] = HiddenLayer(
                                            n_input = self.n_hidden_neurons,
                                            n_output = self.n_categories
                                            )
    def feed_forward(self):
        # feed-forward for training
        #Should a_h be an attribute to hidden layer object, or should it be globally/locally defined in nn?
        a_h = self.hidden_layers[0].node_activation(self.X_data)
        for i in range(1,self.n_hidden_layers+1):
            a_h = self.hidden_layers[i].node_activation(a_h)

        exp_term = np.exp(a_h)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def feed_forward_out(self, X):
        # feed-forward for output
        a_h = self.hidden_layers[0].node_activation_out(X)
        for i in range(1,self.n_hidden_layers+1):
            a_h = self.hidden_layers[i].node_activation_out(a_h)

        exp_term = np.exp(a_h)
        #softmax function
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error = self.probabilities - self.Y_data

        self.hidden_layers[-1].error_hidden = error
        #computing error
        for i in range(len(self.hidden_layers) - 2, -1, -1):

            error = self.hidden_layers[i].error_layer(error, self.hidden_layers[i+1].weights)

        for i in range(len(self.hidden_layers)-1,0,-1):
            self.hidden_layers[i].gradients(self.hidden_layers[i-1].a_h,self.lmbd)

        self.hidden_layers[0].gradients(self.X_data,self.lmbd)

        #updating weights
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].update_weights(self.eta)

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
