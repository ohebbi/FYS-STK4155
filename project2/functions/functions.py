
import numpy as np


def FrankeFunction(x,y):
    """
    Generates Franke's function.
    Input:
    Takes array x and y.
    Output
    Returns array z.
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def generate_data(number_points = 20, plott = True):
    """
    Generates data.
    Input:
    plott = True for plotting.
    Output:
    returns 1D arrays x, y and z (after begin raveled).
    """
    x_data = np.arange(0, 1, 1./number_points)
    y_data = np.arange(0, 1, 1./number_points)

    x, y = np.meshgrid(x_data,y_data)

    z = FrankeFunction(x, y)
    if plott == True:
        plotter(x,y,z)
        plt.savefig('plots/Franke/frankefunction.pdf')

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    print ("x ranges from", 0, "to", 1, "with a total amount of", number_points, "points.")
    print ("y ranges from", 0, "to", 1, "with a total amount of", number_points, "points.")


    eps = np.random.normal(0,1,len(z))
    z += 0.1*eps

    return x, y, z


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