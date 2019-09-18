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




fig = plt.figure()
#ax = fig.gca(projection="3d")

# Make data.
x_data = np.arange(0, 1, 0.05)
y_data = np.arange(0, 1, 0.05)

"""
splitting into training data and test data, to keep the data totally
independent from each other.
"""
#x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.1, shuffle=False,)
#x, y = np.meshgrid(x_train,y_train)

x, y = np.meshgrid(x_data,y_data)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)
#print (z)

#flatten the matrix out
x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)
eps = np.random.normal(0,1,len(z))
z += eps

x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x,y,z, test_size=0.1, shuffle=True)
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


def fitBeta(X,z):

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


    beta = np.linalg.inv(X.T.dot(X) + 0.0*np.identity(len(X.T.dot(X)))).dot(X.T).dot(z)
    return beta

def predictor(X,beta):
    # Now we compute z = Xb

    z_tilde = X.dot(beta)
    return (z_tilde)



# We use now Scikit-Learn's linear regressor for control of our results
#clf = skl.LinearRegression().fit(X, z)
#z_tilde_skl = clf).predict(X)

#print(ztilde-ztilde_skl)

def R2(z_data, z_model):
    return (1 - np.sum((z_data - z_model)**2)/np.sum((z_data - np.mean(z_data))**2))

#print (R2(z,z_tilde))

def MSE(z_data,z_model):
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))
#print (MSE(z,z_tilde))

def crossvalidation(k,x_train, y_train, z_train,beta_len):
    scores_MSE = np.zeros(k)
    scores_R2 = np.zeros(k)
    beta_perfect = np.zeros(beta_len)
    kfold = KFold(n_splits = k, shuffle=True)

    for train_inds, val_inds in kfold.split(x_train):
        xtrain = x_train[train_inds]
        ytrain = y_train[train_inds]
        ztrain = z_train[train_inds]

        xval = x_train[val_inds]
        yval = y_train[val_inds]
        zval = z_train[val_inds]

        #(len(xtrain),len(x), len(xtest))
        Xtrain = designmatrix(xtrain,ytrain, beta_len)
        betatrain = fitBeta(Xtrain,ztrain)

        Xval = designmatrix(xval,yval,beta_len)

        zpred = predictor(Xval,betatrain)
        #print(len(betatrain), len(Xtest))

        beta_perfect += betatrain


    return (beta_perfect/k)


#testing our perfect beta with the test set

scores_R2 = np.zeros(6)
scores_MSE = np.zeros(6)

#Raveling
x_test, y_test = np.meshgrid(x_test, y_test)
x_test = np.ravel(x_test)
y_test = np.ravel(y_test)

z_test = FrankeFunction(x_test, y_test)

t = [0,1,4,8,13,19]
#print (X[:,:5])
j=0
#print(X[:,:5])


for i in t:
    beta_perfect = crossvalidation(5,x_train,y_train,z_train, i+1) #length to beta should be equal to dimension of X
    X = designmatrix(x_test,y_test,len(beta_perfect))

    z_pred = predictor(X,beta_perfect)

    #from beta_perfect
    scores_R2[j] = R2(z_test, z_pred)
    scores_MSE[j] = MSE(z_test, z_pred)
    j += 1
    plt.subplot(2,1,1)
    plt.plot(x_test,z_test)
    plt.subplot(2,1,2)
    plt.title("%f"% i)
    plt.plot(x_test,z_pred)
    plt.show()

print("Mean R2 ",scores_R2)
print("Mean MSE ",scores_MSE)

"""
Task c)
"""
plt.plot([0,1,2,3,4,5],scores_MSE)
plt.show()






"""
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error for sklearn: %f" % mean_squared_error(z, z_tilde))
print("R2 score for sklearn: %f " % r2_score(z, z_tilde))

from sklearn.model_selection import train_test_split
Z_train, Z_test = train_test_split(z_tilde, test_size=0.20)




print (len(Z_train), len(Z_test))

from sklearn.model_selection import KFold
scores_KFold = np.zeros(k)

"""
"""
# Plot the surface.
x_test, y_test = np.meshgrid(x_test,y_test)
z_test = FrankeFunction(x_test,y_test)
print (z_test)
surf = ax.plot_surface(x_test, y_test, z_test, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""
