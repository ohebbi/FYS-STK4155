# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:02:45 2019

@author: mohe9
"""

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

from tqdm import tqdm





"""
splitting into training data and test data, to keep the data totally
independent from each other.
"""
#x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.1, shuffle=False,)
#x, y = np.meshgrid(x_train,y_train)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def fake_data(plott = False):
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
    z += eps

    return x, y, z
    #print (x_train)



def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
#    print('test U')
#    print( (np.transpose(U) @ U - U @np.transpose(U)))
#    print('test VT')
#    print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))
    print(U)
    print(s)
    print(VT)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

def designmatrix(x,y, beta_len):

    x2 = x*x
    y2 = y*y
    x3 = x*x*x
    y3 = y*y*y
    X = np.c_[np.ones((len(x),1)), #0-degree polynomial
                     x, #1-degree polynomial
                     x2,y2,x*y, #2-degree polynomial
                     x3,y3,x*y2,x2*y, #3-degree polynomial
                     x*x3,y*y3,x3*y,x*y3,x2*y2, #4-degree polynomial
                     x3*x2,y3*y2,(x2*x2)*y, x*(y2*y2),x3*y2,x2*y3] #5-degree polynomial

    return X[:,:(beta_len)]


def fitBeta(X,z,lamb=0):

    """
    U, S, V = np.linalg.svd(X)
    print(np.shape(X))
    #print((U),(S),(V))
    svd = U.dot(S)
    svd = svd.dot(V)
    beta = np.linalg.inv(svd.T.dot(svd)).dot(svd.T).dot(z)
    """
    """
    invers = SVDinv(X.T.dot(X))
    beta = invers.dot(X.T).dot(z)
    """


    beta = np.linalg.inv(X.T.dot(X) + lamb*np.identity(len(X.T.dot(X)))).dot(X.T).dot(z)
    return beta

def predictor(X,beta):
    # Now we compute z = Xb

    z_tilde = X.dot(beta)
    return (z_tilde)





def R2(z_data, z_model):
    return (1 - np.sum( (z_data - z_model)**2 ) / np.sum((z_data - np.mean(z_data))**2))

#print (R2(z,z_tilde))

def MSE(z_data,z_model):
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))
#print (MSE(z,z_tilde))


def crossvalidation(k,x_train, y_train, z_train,beta_len,lamb=0):
    scores_MSE = np.zeros(k)
    scores_R2 = np.zeros(k)
    beta_perfect = np.zeros(beta_len)
    kfold = KFold(n_splits = k, shuffle=True)

    #splitting our training data into training- and validation data
    for train_inds, val_inds in kfold.split(x_train):
        xtrain = x_train[train_inds]
        ytrain = y_train[train_inds]
        ztrain = z_train[train_inds]

        xval = x_train[val_inds]
        yval = y_train[val_inds]
        zval = z_train[val_inds]

        #(len(xtrain),len(x), len(xtest))
        Xtrain = designmatrix(xtrain,ytrain, beta_len)
        betatrain = fitBeta(Xtrain,ztrain,lamb)

        Xval = designmatrix(xval,yval,beta_len)

        zpred = predictor(Xval,betatrain)
        #print(len(betatrain), len(Xtest))

        beta_perfect += betatrain


    return (beta_perfect/k)


#testing our perfect beta with the test set


#splitte inn i x,y,z senere for aa slippe aa ta inn flere variabler??

#
def tester(x_train,y_train,z_train,x_test,y_test,z_test,lamb):

    #Raveling
    x_test, y_test = np.meshgrid(x_test, y_test)
    x_test = np.ravel(x_test)
    y_test = np.ravel(y_test)

    z_test = FrankeFunction(x_test, y_test)

    #polynomial of degree 0 to 5
    #t1 = [0,1,4,8,13,19]
    j=0

    #polynomial from factor 1 to 19
    t = np.linspace(0,19,20)

    scores_R2 = np.zeros(len(t))
    scores_MSE = np.zeros(len(t))
    bias = np.zeros(len(t))
    variance = np.zeros(len(t))

    for i in (t):
        beta_perfect = crossvalidation(5,x_train,y_train,z_train, int(i), lamb) #length to beta should be equal to dimension of X
        X = designmatrix(x_test,y_test,len(beta_perfect))

        z_pred = X @ beta_perfect

        #from beta_perfect
        scores_R2[j] = R2(z_test, z_pred)
        scores_MSE[j] = MSE(z_test, z_pred)
        square = z_test - np.mean(z_pred)
        bias[j] = np.mean( (square)**2 )
        variance[j] = np.mean( np.var(z_pred))

        j += 1


    return variance,scores_MSE,bias


    #plt.show()



def ridge_regression(x_train, x_test, y_train, y_test,z_train, z_test):

    #Why no work for lambda=-0.25 ish????
    l = np.linspace(0,2,20)
    #nice plots
    color=iter(plt.cm.rainbow(np.linspace(1,0,len(l))))
    t = np.linspace(0,19,20)
    for lamb in tqdm(l):
        variance, scores_MSE, bias = tester(x_train,y_train,z_train,x_test,y_test,z_test,lamb)
        c = next(color)
        #plt.plot(t,(scores_MSE), ',')
        plt.plot(t,(bias), '-',c=c)
        #plt.plot(t,(bias),',')

    plt.legend(l)
    plt.title("")
    plt.show()


def lasso_regression(x_train, x_test, y_train, y_test,z_train, z_test):

    l = np.linspace(0.1,2,20)
    #beta_len = antall dim i X
    t = np.linspace(1,19,20)



    X_train=designmatrix(x_train,y_train, 19) #dim+1???
    #print (X_train)
    #print (z_train)
    clf_lasso = skl.Lasso(alpha=0.1)
    method = clf_lasso.fit(X_train,z_train)

    X_test = designmatrix(x_test,y_test,19)
    print (method.score(X_train, z_train))
    print (method.score(X_test, z_test))

    """
    for dim in t:
        for lamb in (l):

            X_train=designmatrix(x_train,y_train, int(dim+1)) #dim+1???
            clf_lasso = skl.Lasso(alpha=lamb)
            method = clf_lasso.fit(X_train,z_train)

            print (method.score(X_train, z_train))
    """
def main():

    #exercise a)
    x,y,z = fake_data()
    x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)


    # We use now Scikit-Learn's linear regressor for control of our results
    #clf = skl.LinearRegression().fit(X, z)
    #z_tilde_skl = clf).predict(X)

    #print(ztilde-ztilde_skl)


    #exercise c)
    #K-fold CV lamb=0
    t = np.linspace(0,19,20)
    variance, scores_MSE, bias = tester(x_train,y_train,z_train,x_test,y_test,z_test,lamb=0)

    #print("Mean R2 ",scores_R2)
    print("Mean MSE ",scores_MSE)
    print("Bias^2", bias)
    print("Variance", bias)


    plt.plot(t,scores_MSE,'y')
    plt.plot(t,bias, 'b', )

    plt.plot(t,variance,'r')

    #How MSE should look like
    plt.plot(t,variance+bias)
    plt.legend(["MSE","Bias^2","Variance"])
    plt.show()


    #exercise d)
    #ridge_regression(x_train, x_test, y_train, y_test,z_train, z_test)

    #exercise e)
    #lasso_regression(x_train, x_test, y_train, y_test,z_train, z_test)


if __name__ == '__main__':
    main()
