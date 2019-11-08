import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
        #plt.savefig('plots/Franke/frankefunction.pdf')

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    print ("x ranges from", 0, "to", 1, "with a total amount of", number_points**2, "points.")
    print ("y ranges from", 0, "to", 1, "with a total amount of", number_points**2, "points.")


    eps = np.random.normal(0,1,len(z))
    z += 0.01*eps

    return x, y, z

def R2(z_data, z_model):
    """
    Function:
    Finds the R2-level for a given model and approximation.
    Input:
    Takes an array z_data and z_model.
    Output:
    Returns a scalar.
    """
    return (1 - np.sum( (z_data - z_model)**2 ) / np.sum((z_data - np.mean(z_data))**2))

def plotter(x,y,z):
    """
    Function:
    Generates a three dimensional plot.
    Input:
    Takes an array x, y and z.
    Output:
    Gives a plot.
    """
    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    for angle in range(0,150):
        ax.view_init(40,angle)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5);

def MSE(z_data,z_model):
    """
    Function:
    Finds the mean square error for a given model and approximation.
    Input:
    Takes an array z_data and z_model.
    Output:
    Returns a scalar.
    """
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))

def find_designmatrix(x,y, polygrad=5):
    """
    Function:
    Generates the designmatrix.
    Input:
    Takes an array x and y and a polynomial degree.
    Output:
    Returns a multidimensional array (designmatrix).
    """
    x2 = x*x
    y2 = y*y
    x3 = x*x*x
    y3 = y*y*y

    if (polygrad<1):
        raise ValueError ("error! polygrad is less than 1!!")

    if polygrad == 1:
        X = np.c_[np.ones((len(x),1)),x, y] #3
    elif (polygrad == 2):
        X = np.c_[np.ones((len(x),1)), #0-degree polynomial
                     x, y, #1-degree polynomial
                     x2,y2,x*y] #2-degree polynomial #6
    elif polygrad == 3:
        X = np.c_[np.ones((len(x),1)), #0-degree polynomial
                         x, y, #1-degree polynomial
                         x2,y2,x*y, #2-degree polynomial
                         x3,y3,x*y2,x2*y] #3-degree polynomial #10
    elif polygrad == 4:
        X = np.c_[np.ones((len(x),1)), #0-degree polynomial
                         x, y, #1-degree polynomial
                         x2,y2,x*y, #2-degree polynomial
                         x3,y3,x*y2,x2*y, #3-degree polynomial
                         x*x3,y*y3,x3*y,x*y3,x2*y2] #4-degree polynomial #15

    elif polygrad ==5:
        X = np.c_[np.ones((len(x),1)), #0-degree polynomial
                     x, y, #1-degree polynomial
                     x2,y2,x*y, #2-degree polynomial
                     x3,y3,x*y2,x2*y, #3-degree polynomial
                     x*x3,y*y3,x3*y,x*y3,x2*y2, #4-degree polynomial
                     x3*x2,y3*y2,(x2*x2)*y, x*(y2*y2),x3*y2,x2*y3] #5-degree polynomial #21

    #General formula to avoid hardcoding 'too' much.
    elif (polygrad > 5):
        X = np.zeros( (len(x), int(0.5*(polygrad + 2)*(polygrad + 1)) ) )
        poly = 0
        for i in range(int(polygrad) + 1):
            for j in range(int(polygrad) + 1 - i):
                X[:,poly] = np.squeeze((x**i)*(y**j))
                poly += 1
    return X

def inv_sigmoid(y):
    """
    Function:
    Scales the input back after being scaled with sigmoid
    Input:
    Takes an input scalar, list, array or matrix
    Output:
    Returns the inverse version of the input
    """
    if type(y)==int:
        if y == 1:
            msg = "Inverse of sigmoid is not defined for y=1"
            raise TypeError(msg)
    if type(y)==list:
        for i in y:
            if i == 1:
                msg = "Inverse of sigmoid is not defined for y=1"
                raise TypeError(msg)
    return np.log(y/(1-y))

def sigmoid(x):
    """
    Function:
    Scales the input after sigmoid
    Input:
    Takes an input scalar, list, array or matrix
    Output:
    Returns the scaled version of the input
    """
    return 1./(1.+np.exp(-x))

def learning_schedule(t):
    """
    Function:
    Learning schedule for stochastic gradient method
    Input:
    Takes an input scalar, list, array or matrix
    Output:
    Returns the scaled version of the input
    """
    t0, t1 = 1, 50
    return t0/(t+t1)

def to_categorical_numpy(integer_vector):
    """
    Function:
    Onehots a vector
    Input:
    Takes an array
    Output:
    Returns two-dim. array
    """
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

def accuracy_score_numpy(Y_test, Y_pred):
    """
    Function:
    Returns the accuracy score
    Input:
    Takes an input array Y_test and a corresponding prediction Y_pred
    Output:
    Returns a scalar.
    """
    if len(Y_test) != len(Y_pred):
        msg = ("The length of Y_test and Y_pred is not equal", Y_test.shape, Y_pred.shape)
        raise ValueError(msg)
    return np.sum(Y_test == Y_pred) / len(Y_test)

def best_curve(y,summ):
    """
    Function:
    Finds the ideal curve for a cumulative gain curve
    Input:
    Takes an array y and the class 0 or 1 as summ
    Output:
    Returns array x and array y3 
    """
	defaults = sum(y == summ)
	total = len(y)
	x = np.linspace(0, 1, total)
	y1 = np.linspace(0, 1, defaults)
	y2 = np.ones(total-defaults)
	y3 = np.concatenate([y1,y2])
	return x, y3
