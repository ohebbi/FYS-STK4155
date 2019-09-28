# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:02:45 2019

@author: oheb
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import cm

from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split

from tqdm import tqdm


from functions import *


def main(x,y,z):

    #exercise a)
    #x,y,z = generate_data()

    #if overfit: activate next line
    #x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)



    # We use now Scikit-Learn's linear regressor for control of our results
    #clf = skl.LinearRegression().fit(X, z)
    #z_tilde_skl = clf).predict(X)

    #print(ztilde-ztilde_skl)
    #exercise b)
    """
    5-fold crossvalidation OLS
    """
    # Polynomial degree
    degrees = np.linspace(1,10,10)

    bias = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    test_MSE_OLS = np.zeros(len(degrees))
    test_R2 = np.zeros(len(degrees))

    train_MSE = np.zeros(len(degrees))
    train_R2 = np.zeros(len(degrees))

    #x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)

    #    print (x_train)

    """
    OLS regression
    """

    k = 5 #cross fold
    for polygrad in degrees:

        j = int(polygrad) - 1
        scores, betas = bias_variance(x,y,z,polygrad,k, regressiontype='OLS')
        train_MSE[j] = scores[0]
        train_R2[j] = scores[1]

        test_MSE_OLS[j] = scores[2]
        bias[j] = scores[3]
        variance[j] = scores[4]

        #print('Bias^2:', test_bias[j])
        #print('Var:', test_variance[j])
        #print('{} >= {} + {} = {}'.format(test_MSE[j],test_bias[j], test_variance[j], test_bias[j]+test_variance[j]))

    plt.plot(degrees,test_MSE_OLS)
    plt.plot(degrees,variance)
    plt.plot(degrees,bias)

    plt.legend(["test_MSE","variance", "bias"])
    plt.title("OLS regression")
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.show()

    plt.plot(degrees,train_MSE)
    plt.plot(degrees,test_MSE_OLS)
    plt.legend(["train_MSE","test_MSE"])
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.title("OLS regression")
    plt.show()

    """
    Ridge_regression
    """

    """
    First running CV for finding best lambda with lowest MSE.
    """

    nlambdas = 10
    lambdas = np.logspace(-3,0,nlambdas)

    color=iter(cm.rainbow(np.linspace(1,0,nlambdas)))

    for lamb in lambdas:

        test_MSE_Ridge = Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='Ridge')

        c = next(color)
        plt.plot(degrees,test_MSE_Ridge, c=c)
        plt.legend(lambdas)
        plt.xlabel("Complexity of model (the degree of the polynomial)")
        plt.ylabel("Error")
        plt.title("Ridge_MSE for different lambda-values")
    plt.show()

    """
    Then calculate the bias-variance with the best lambda.
    As an example we now use "best" lambda = 0.001
    """


    lamb = 0.001

    test_MSE_Ridge, Bias, Variance, betas = Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='Ridge')

    plt.plot(degrees,test_MSE_Ridge)
    plt.legend(["lamb = 0.001"])
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.title("Ridge_MSE for best lambda-value")
    plt.show()


    """
    Lasso_regression
    """

    """
    It doesn't like large lambda values, so try lambda in the magnitude of 10^-6.
    But be aware that you may get a warning saying you:
    "ConvergenceWarning:
    Objective did not converge. You might want to increase the number of iterations.
    Fitting data with very small alpha may cause precision problems."
    """

    """
    First running CV for finding best lambda with lowest MSE.
    """

    #nlambdas = 4
    #lambdas = np.logspace(-6,-2,nlambdas)


    color=iter(cm.rainbow(np.linspace(1,0,nlambdas)))

    for lamb in tqdm(lambdas):

        test_MSE_LASSO= Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='Lasso')


        c = next(color)
        plt.plot(degrees,test_MSE_LASSO, c=c)
        plt.legend(lambdas)
        plt.xlabel("Complexity of model (the degree of the polynomial)")
        plt.ylabel("Error")
        plt.title("Lasso_MSE for different lambda-values")
    plt.show()


    """
    Then calculate the bias-variance with the best lambda.
    As an example we now use "best" lambda = 0.1
    """
    lamb = 0.1

    test_MSE_LASSO, Bias, Variance, betas = Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='Lasso')

    plt.plot(degrees,test_MSE_LASSO)
    plt.legend(["lamb = 0.1"])
    plt.title("Lasso_MSE for best lambda-value")
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.show()


    """
    plt.plot(t,train_MSE,'r')
    plt.plot(t,test_MSE,'b')
    plt.legend(["train MSE","test MSE"])
    plt.show()
    """

    """
    Terrain data
    """
    """
    x,y,z = terrain_data()
    x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)

    test_MSE = np.zeros(len(t))
    test_bias = np.zeros(len(t))
    test_variance = np.zeros(len(t))

    train_MSE = np.zeros(len(t))
    train_bias = np.zeros(len(t))
    train_variance = np.zeros(len(t))


    k = 5 #cross fold
    for polygrad in tqdm(t):

        j = int(polygrad) - 1
        train_MSE[j], train_bias[j], train_variance[j], beta = crossvalidation(x_train,y_train,z_train, k, polygrad, regressiontype = 'OLS') #length to beta should be equal to dimension of X
        X = find_designmatrix(x_test,y_test, polygrad)

        z_pred = X.dot(beta)

        #scores_R2[j] = R2(z_test, z_pred)
        test_MSE[j] = MSE(z_test, z_pred)
        square = z_test - np.mean(z_pred)
        test_bias[j] = np.mean( (square)**2 )
        test_variance[j] = np.mean( np.var(z_pred))
    plt.plot(t,train_MSE)
    plt.plot(t,train_variance)
    plt.plot(t,train_bias)
    plt.legend(["train MSE","train variance", "train bias"])
    plt.show()
    plt.plot(t,test_MSE)
    plt.plot(t,test_variance)
    plt.plot(t,test_bias)
    plt.plot(t,test_variance+test_bias)
    plt.legend(["test MSE","test variance", "test bias","bias + variance"])
    plt.show()

    plt.plot(t,train_MSE,'r')
    plt.plot(t,test_MSE,'b')
    plt.legend(["train MSE","test MSE"])
    plt.show()
    """
    #exercise d)
    #ridge_regression(x_train, x_test, y_train, y_test,z_train, z_test)

    #exercise e)
    #lasso_regression(x_train, x_test, y_train, y_test,z_train, z_test)
def bootstrap_main(x,y,z):
    degrees = np.linspace(1,12,12)
    error_test = bootstrap(x,y,z,degrees,"OLS")
    print (error_test)
    plt.plot(degrees,error_test)
    plt.show()

if __name__ == '__main__':
    x,y,z = generate_data()
    main(x,y,z)
    #bootstrap_main(x,y,z)

    #print("======================================\nTerrain data\n======================================")
    #x,y,z = terrain_data()
    #main(x,y,z)
