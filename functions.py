import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split


from regression_functions import *

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_data(plott = False):
    # Make data.
    x_data = np.arange(0, 1, 0.05)
    y_data = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x_data,y_data)

    z = FrankeFunction(x, y)


    if plott == True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
                       # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)
    eps = np.random.normal(0,1,len(z))
    z += 0.1*eps

    return x, y, z
    #print (x_train)



def find_designmatrix(x,y, polygrad=5):

    x2 = x*x
    y2 = y*y
    x3 = x*x*x
    y3 = y*y*y
    if (polygrad>5):
        print ("error! polygrad is bigger than five!!")
        exit(1)
    if (polygrad<1):
        print ("error! polygrad is less than 1!!")
        exit(1)

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
    return X

def R2(z_data, z_model):
    return (1 - np.sum( (z_data - z_model)**2 ) / np.sum((z_data - np.mean(z_data))**2))

def MSE(z_data,z_model):
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))

#We are approximating sigma to be equal MSE
def confidence_interval(beta, MSE):
    sigma = np.sqrt(MSE)
    mean_beta = 0
    for i in beta:
        mean_beta += i
    print ("confidence interval is from %2.4f to %2.4f." %
            (mean_beta-sigma*1.96, mean_beta+sigma*1.96))

def OLS(X,z):

    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    return beta

def ridge_regression(X,z,lamb):

    beta = np.linalg.inv(X.T.dot(X) + lamb*np.identity(len(X.T.dot(X)))).dot(X.T).dot(z)
    return beta

def crossvalidation(x_train, y_train, z_train, k, polygrad, regressiontype = 'OLS',lamb=0):

    scores_MSE = np.zeros(k)
    bias = np.zeros(k)
    variance = np.zeros(k)
    scores_R2 = np.zeros(k)

    #finding correct length of beta with the beautiful dummy variabe
    dummy_variable = find_designmatrix(np.zeros(1),np.zeros(1),polygrad)
    beta_perfect = np.zeros(len(dummy_variable[0]))

    kfold = KFold(n_splits = k, shuffle=True)

    #splitting our training data into training- and validation data
    i =0
    for train_inds, val_inds in kfold.split(x_train):
        xtrain = x_train[train_inds]
        ytrain = y_train[train_inds]
        ztrain = z_train[train_inds]

        xval = x_train[val_inds]
        yval = y_train[val_inds]
        zval = z_train[val_inds]

        #(len(xtrain),len(x), len(xtest))
        Xtrain = find_designmatrix(xtrain,ytrain, polygrad)

        if regressiontype == 'OLS':
            betatrain = OLS(Xtrain,ztrain)
        elif regressiontype == 'Ridge':
            betatrain = ridge_regression(Xtrain, ztrain, lamb)

        else:
            print ("regression-type is lacking input!")
            exit(1)



        """ To be implemented!
        elif regressiontype = 'Lasso':
            betratrain = lasso()
        """

        Xval = find_designmatrix(xval,yval,polygrad)

        zpred = Xval @ betatrain
        #print(len(betatrain), len(Xtest))


        scores_MSE[i] =  MSE(zval,zpred)
        square = (zval - np.mean(zpred,axis=1,keepdims=True))**2
        bias[i] = np.mean((square))
        variance[i] = np.mean( np.var(zpred))
        scores_R2[i] = R2(zval,zpred)
        #print (len(beta_perfect), len(betatrain))
        beta_perfect += betatrain
        i += 1

    estimated_MSE = np.mean(scores_MSE)
    estimated_bias = np.mean((bias))
    estimated_variance = np.mean(variance)
    return (estimated_MSE, estimated_bias, estimated_variance, (beta_perfect/float(k)))
