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


from regression_functions import *
from functions import *


def main():

    #exercise a)
    x,y,z = generate_data()

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
    t = np.linspace(3,10,8)

    bias = np.zeros(len(t))
    variance = np.zeros(len(t))

    test_MSE = np.zeros(len(t))
    test_R2 = np.zeros(len(t))

    train_MSE = np.zeros(len(t))
    train_R2 = np.zeros(len(t))

    #x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)

    #    print (x_train)

    k = 5 #cross fold
    for polygrad in t:

        j = int(polygrad) - 3
        scores = bias_variance(x,y,z,polygrad,k, regressiontype='OLS')

        train_MSE[j] = scores[0]
        train_R2[j] = scores[1]

        test_MSE[j] = scores[2]
        bias[j] = scores[3]
        variance[j] = scores[4]

        #print('Bias^2:', test_bias[j])
        #print('Var:', test_variance[j])
        #print('{} >= {} + {} = {}'.format(test_MSE[j],test_bias[j], test_variance[j], test_bias[j]+test_variance[j]))

    plt.plot(t,test_MSE)
    plt.plot(t,variance)
    plt.plot(t,bias)

    plt.legend(["test_MSE","variance", "bias"])
    plt.show()

    plt.plot(t,train_MSE)
    plt.plot(t,test_MSE)
    plt.show()

    """
    ridge_regression
    """

    nlambdas = 14
    lambdas = np.logspace(-3,1,nlambdas)

    color=iter(cm.rainbow(np.linspace(1,0,nlambdas)))

    for lamb in tqdm(lambdas):
        k = 5 #cross fold
        for polygrad in t:

            j = int(polygrad) - 3
            scores, betas = k_fold_cross_validation(x,y,z,k,polygrad, regressiontype='OLS')

            train_MSE[j] = scores[0]
            train_R2[j] = scores[1]



        c = next(color)
        plt.plot(t,train_MSE,c=c)
        plt.legend(lambdas)
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


if __name__ == '__main__':
    main()
