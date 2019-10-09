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
from functions.functions import *
import seaborn as sns
from pandas import DataFrame

def main(x,y,z):

    # Polynomial degree
    degrees = np.linspace(1,10,10)

    # Storing the bias & variance
    bias = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    # Storing the MSE of both the test and train data
    test_MSE_OLS = np.zeros(len(degrees))
    train_MSE = np.zeros(len(degrees))

    # Storing the R2-scores
    R2_OLS = np.zeros(len(degrees))

    # Storing the confidence interval of the regression coeficcient beta
    CI_OLS = np.zeros((len(degrees),3))

    # Storing the values of the regression coefficient beta for OLS
    betas_OLS = {}

    """
    OLS-regression
    """

    k = 5 #cross fold
    for polygrad in degrees:

        j = int(polygrad) - 1
        scores, betas_OLS[int(polygrad)] = bias_variance(x,y,z,polygrad,k, regressiontype='OLS')

        train_MSE[j] = scores[0]
        R2_OLS[j] = scores[1]

        test_MSE_OLS[j] = scores[2]
        bias[j] = scores[3]
        variance[j] = scores[4]
        CI_OLS[j] = scores[5]

    """
    Plotting the confidence interval of the regression coefficient beta for OLS
    """

    #plt.plot(degrees,CI_OLS[:,0])
    #plt.plot(degrees,CI_OLS[:,1])
    #plt.plot(degrees,CI_OLS[:,2])
    #plt.legend(["low","mean", "highest beta"])
    #plt.ylabel("mean Beta")
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    ##plt.show()


    #fig, ax = plt.subplots()

    #ax.set_ylim(-0.5,0.5)
    #(_, caps, _) = #plt.errorbar(degrees, CI_OLS[:,1], yerr=(CI_OLS[:,0]-CI_OLS[:,1]), fmt='o', markersize=4, capsize=6,label='jepp')

    #for cap in caps:
    #    cap.set_markeredgewidth(1)
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("mean Beta")
    ##plt.show()


    """
    The MSE of the test and train data plotted against model complexity for OLS
    """


    #plt.plot(degrees,test_MSE_OLS)
    #plt.plot(degrees,train_MSE)
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("MSE")
    #plt.title("OLS Mean Square Error")
    #plt.legend(["test MSE","train MSE"])
    ##plt.show()


    """
    The MSE, bias and variance plotted against model complexity for OLS
    """

    #plt.plot(degrees,test_MSE_OLS)
    #plt.plot(degrees,variance)
    #plt.plot(degrees,bias)
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.title("OLS - Bias-variance tradeoff")
    #plt.ylabel("Error")
    #plt.plot(degrees,variance+bias,'o')
    #plt.legend(["MSE","variance", "bias$^2$","bias$^2$+variance"])
    ##plt.show()


    """
    Ridge-regression
    """

    """
    Comparing different values of lambda with respect to MSE
    """

    nlambdas = 10
    #lambdas = np.logspace(-3,-2,nlambdas)
    lambdas = np.linspace(1e-5,1e-2,nlambdas)
    color=iter(cm.rainbow(np.linspace(1,0,nlambdas)))
    heatmap_mse = np.zeros((nlambdas,len(degrees)))

    i=0
    for lamb in lambdas:

        test2_MSE = Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='Ridge')
        heatmap_mse[i] = test2_MSE
        i += 1
        c = next(color)
        #plt.plot(degrees,test2_MSE, c=c)
        #plt.legend(lambdas)
        #plt.xlabel("Complexity of model (the degree of the polynomial)")
        #plt.ylabel("Error")
        #plt.title("Ridge_MSE for different lambda-values")
    ##plt.show()
    df = DataFrame(heatmap_mse, index = lambdas, columns = degrees)
    #fig = sns.heatmap(df, cmap="YlGnBu", yticklabels=df.index.values.round(5),fmt='.2g',cbar_kws={'label': 'Mean square error'})
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("values for $\lambda$")
    #plt.title("Heatmap for Ridge-regression")
    ##plt.show()

    """
    Choosing a lambda to do further regression analysis
    As an example we choose lambda = 0.001
    """

    """
    The MSE of the test and train data plotted against model complexity for Ridge
    """

    lamb_Ridge = 1e-3

    test_MSE_Ridge, R2_Ridge, Bias, Variance, CI_Ridge, betas_Ridge,train_MSE_Ridge = Best_Lambda(x, y, z, degrees, k, lamb_Ridge, regressiontype='Ridge')

    #plt.plot(degrees,test_MSE_Ridge)
    #plt.plot(degrees,train_MSE_Ridge)
    #plt.legend(["test MSE","train MSE"])
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("Error")
    #plt.title("Ridge_MSE for $\lambda=%g $"%(lamb_Ridge))
    ##plt.show()


    """
    The MSE, bias and variance plotted against model complexity for Ridge
    """

    #plt.plot(degrees,test_MSE_Ridge)
    #plt.plot(degrees,Variance)
    #plt.plot(degrees,Bias)
    #plt.title("Ridge - Bias-variance tradeoff with $\lambda=1e-3$")
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("Error")
    #plt.plot(degrees,Variance+Bias,'o')
    #plt.legend(["test_MSE","variance", "bias$^2$","bias$^2$+variance"])
    ##plt.show()


    """
    Plotting the confidence interval of the regression coefficient beta for Ridge
    """

    #plt.plot(degrees,CI_Ridge[:,0])
    #plt.plot(degrees,CI_Ridge[:,1])
    #plt.plot(degrees,CI_Ridge[:,2])
    #plt.legend(["low","mean", "high"])
    #plt.ylabel("mean Beta")
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    ##plt.show()


    #fig, ax = plt.subplots()

    #ax.set_ylim(-0.6,0.6)
    #(_, caps, _) = plt.errorbar(degrees, CI_Ridge[:,1], yerr=(CI_Ridge[:,0]-CI_Ridge[:,1]), fmt='o', markersize=4, capsize=6)

    #for cap in caps:
    #    cap.set_markeredgewidth(1)
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("mean Beta")
    ##plt.show()


    """
    LASSO-regression
    """

    """
    Comparing different values of lambda with respect to MSE
    """

    nlambdas = 11
    lambdas = np.linspace(1e-6,1e-2,nlambdas)
    color=iter(cm.rainbow(np.linspace(1,0,nlambdas)))
    array_LASSO = np.zeros((nlambdas,len(degrees)))
    i=0

    for lamb in tqdm.tqdm(lambdas):

        test_MSE_LASSO = Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='Lasso')
        array_LASSO[i] = test_MSE_LASSO
        i += 1
        c = next(color)
        #plt.plot(degrees,test_MSE_LASSO, c=c)
        #plt.legend(lambdas)
        #plt.xlabel("Complexity of model (the degree of the polynomial)")
        #plt.ylabel("Error")
        #plt.title("LASSO_MSE for different $lambda$-values")
    ##plt.show()


    df_LASSO = DataFrame(array_LASSO, index = lambdas, columns = degrees)
    #fig = sns.heatmap(df_LASSO, cmap="YlGnBu", yticklabels=df.index.values.round(5),fmt='.4g',cbar_kws={'label': 'Mean square error'})
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("values for $\lambda$")
    #plt.title("Heatmap for LASSO-regression")
    ##plt.show()




    """
    Choosing a lambda to do further regression analysis
    As an example we choose lambda = 0.0001
    """

    """
    The MSE of the test and train data plotted against model complexity for LASSO
    """

    lamb_LASSO = 1e-4

    test_MSE_LASSO,R2_LASSO, Bias, Variance, CI_LASSO, betas_LASSO, train_MSE_LASSO = Best_Lambda(x, y, z, degrees, k, lamb_LASSO, regressiontype='Lasso')

    #plt.plot(degrees,test_MSE_LASSO)
    #plt.plot(degrees,train_MSE_LASSO)
    #plt.legend(["test MSE","train MSE"])
    #plt.title("Lasso_MSE for $\lambda=%g$"%(lamb_LASSO))
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("Error")
    ##plt.show()


    """
    The MSE, bias and variance plotted against model complexity for LASSO
    """

    #plt.plot(degrees,test_MSE_LASSO)
    #plt.plot(degrees,Variance)
    #plt.plot(degrees,Bias)
    #plt.title("LASSO - Bias-variance tradeoff with $\lambda=%0.4f$"%(lamb_LASSO))
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("Error")
    #plt.plot(degrees,Variance+Bias,'o')
    #plt.legend(["test_MSE","variance", "bias$^2$","bias$^2$+variance"])
    ##plt.show()




    """
    Plotting the confidence interval of the regression coefficient beta for LASSO
    """


    #plt.plot(degrees,CI_LASSO[:,0])
    #plt.plot(degrees,CI_LASSO[:,1])
    #plt.plot(degrees,CI_LASSO[:,2])
    #plt.legend(["low","mean", "highest beta"])
    #plt.ylabel("mean Beta")
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    ##plt.show()


    #fig, ax = plt.subplots()

    #ax.set_ylim(-0.5,0.5)
    #(_, caps, _) = plt.errorbar(degrees, CI_LASSO[:,1], yerr=(CI_LASSO[:,0]-CI_LASSO[:,1]), fmt='o', markersize=4, capsize=6,label='jepp')

    #for cap in caps:
    #    cap.set_markeredgewidth(1)
    #plt.title("Confidence interval for beta with $\lambda=%f$"%(lamb_LASSO))
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("mean Beta")
    #plt.show()



    """
    The R2-score plotted against model complexity for OLS, Ridge and LASSO
    """

    #plt.plot(degrees,R2_OLS)
    #plt.plot(degrees,R2_Ridge)
    #plt.plot(degrees,R2_LASSO)
    #plt.legend(["OLS","Ridge: $\lambda=%0.3f$"%lamb_Ridge,"LASSO: $\lambda=%0.4f$"%lamb_LASSO])
    #plt.xlabel("Complexity of model (the degree of the polynomial)")
    #plt.ylabel("R2 Score")
    #plt.title("R2 analysis for regression types")
    #plt.show()

    # Printing out a table of the R2-scores of the different regression methods
    R2 = {"Poly-Degrees":degrees,"OLS R2":R2_OLS, "Ridge R2": R2_Ridge, "LASSO R2":R2_LASSO}
    R2 = DataFrame(R2)
    print(R2)



    """
    To run the main function one needs to call it and specify which data one wants to use
    If one wants the Franke function you need to use the function generate_data from functions
    Example :

    x,y,z = generate_data()

    For the terrain data one needs to call for the terrain_data from functions
    and specify how many points you want to run
    Example:

    x,y,z = terrain_data(skip_nr_points=10)
    """
    return [np.amax(R2_OLS),np.amax(R2_Ridge),np.amax(R2_LASSO)],[np.amin(test_MSE_OLS), np.amin(test_MSE_Ridge), np.amin(test_MSE_LASSO)]
if __name__ == '__main__':
    #inputs = [20,25, 30,35,40,45,50,55,60,65,70,75,80,85,90,95,100] #Franke
    inputs = [50,45,40,35,30,25,20,15,10] #terrain
    sns.set()
    R2_OLS = np.zeros(len(inputs))
    R2_Ridge = np.zeros(len(inputs))
    R2_LASSO = np.zeros(len(inputs))

    MSE_OLS = np.zeros(len(inputs))
    MSE_Ridge = np.zeros(len(inputs))
    MSE_LASSO = np.zeros(len(inputs))

    datapoints = np.zeros(len(inputs))

    i = 0
    for j in inputs:
        x,y,z = terrain_data(j, plott=False)
        datapoints[i] = (len(x)+len(y))/2
        scores_R2, scores_MSE = main(x,y,z)

        R2_OLS[i] = scores_R2[0]
        R2_Ridge[i] = scores_R2[1]
        R2_LASSO[i] = scores_R2[2]

        MSE_OLS[i] = scores_MSE[0]
        MSE_Ridge[i] = scores_MSE[1]
        MSE_LASSO[i] = scores_MSE[2]
        i += 1

    plt.subplot(2,1,1)
    plt.title("R2 as function of data points")
    plt.ylabel("R2-score")

    plt.plot(datapoints, R2_OLS)
    plt.plot(datapoints, R2_Ridge)
    plt.plot(datapoints, R2_LASSO)
    plt.legend(["OLS", "Ridge", "LASSO"])

    plt.subplot(2,1,2)
    plt.title("MSE as function of data points")

    plt.ylabel("MSE")
    plt.xlabel("number of data points")
    plt.plot(datapoints, MSE_OLS)
    plt.plot(datapoints, MSE_Ridge)
    plt.plot(datapoints, MSE_LASSO)
    plt.legend(["OLS", "Ridge", "LASSO"])

    print ("Highest R square score for OLS:", np.amax(R2_OLS))
    print ("Highest R square score for Ridge:", np.amax(R2_Ridge))
    print ("Highest R square score for LASSO:", np.amax(R2_LASSO))
    print ("Lowest MSE score for OLS:", np.amin(MSE_OLS))
    print ("Lowest MSE score for Ridge:", np.amin(MSE_Ridge))
    print ("Lowest MSE score for LASSO:", np.amin(MSE_LASSO))
    plt.show()

    #bootstrap_main(x,y,z)


    print("======================================\nTerrain data\n======================================")
    #x,y,z = terrain_data(skip_nr_points=20)
    #main(x,y,z)
