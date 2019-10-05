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
    betas_OLS = {}

    """
    OLS regression
    """

    k = 5 #cross fold
    for polygrad in degrees:

        j = int(polygrad) - 1
        scores, betas_OLS[int(polygrad)] = bias_variance(x,y,z,polygrad,k, regressiontype='OLS')

        train_MSE[j] = scores[0]
        test_R2[j] = scores[1]

        test_MSE_OLS[j] = scores[2]
        bias[j] = scores[3]
        variance[j] = scores[4]

    X = find_designmatrix(x,y,2)
    beta = betas_OLS[2]
    z_OLS = X @ beta
    #z_Ridge = X @ betas_Ridge['7']
    #z_LASSO = X @ betas_LASSO['7']
    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))
    z = np.reshape(z,(len(z),1))

    plotter(x, y, z_OLS)

    x, y = np.meshgrid(x,y)
    #z_OLS = np.reshape(z_OLS,(len(z_OLS)),1)

    #print (x, y, z_OLS)
    #plotter(x, y, z_OLS)

    exit(1)



    plt.legend(["test_MSE","variance", "bias"])
    plt.title("OLS regression Bias-Variance Tradeoff")
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.show()

    plt.plot(degrees,train_MSE)
    plt.plot(degrees,test_MSE_OLS)
    plt.legend(["train_MSE","test_MSE"])
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.title("OLS regression MSE")
    plt.show()

    plt.plot(degrees,test_R2)
    plt.legend(["test_R2"])
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.title("OLS regression R2-score")
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

    test_MSE_Ridge, R2_Ridge, Bias, Variance, CI, betas_Ridge = Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='Ridge')

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

    test_MSE_LASSO, R2_LASSO, Bias, Variance, CI, betas_LASSO = Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='Lasso')

    plt.plot(degrees,test_MSE_LASSO)
    plt.legend(["lamb = 0.1"])
    plt.title("Lasso_MSE for best lambda-value")
    plt.xlabel("Complexity of model (the degree of the polynomial)")
    plt.ylabel("Error")
    plt.show()

    X = find_designmatrix(x_test,y_test)
    z_OLS = X @ betas_OLS[7]
    #z_Ridge = X @ betas_Ridge['7']
    #z_LASSO = X @ betas_LASSO['7']

    z_test = np.reshape(z_test,(len(z_test),1))

    plotter(x_test, y_test, z_test)
    plotter(x_test, y_test, z_OLS)
    #plotter(x_test, y_test, z_Ridge)
    #plotter(x_test, y_test, z_LASSO)


def bootstrap_main(x,y,z):
    degrees = np.linspace(1,12,12)
    error_test = bootstrap(x,y,z,degrees,"OLS")
    print (error_test)
    plt.plot(degrees,error_test)
    plt.show()

if __name__ == '__main__':
    x,y,z = generate_data()
    #main(x,y,z)
    #bootstrap_main(x,y,z)

    print("======================================\nTerrain data\n======================================")
    x,y,z = terrain_data(skip_nr_points=20)
    #main(x,y,z)
