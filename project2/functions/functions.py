import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))
def learning_schedule(t):
    t0, t1 = 1, 50
    return t0/(t+t1)

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector
def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)
