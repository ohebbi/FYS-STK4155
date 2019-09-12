# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:02:45 2019

@author: mohe9
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import sklearn.linear_model as skl




fig = plt.figure()
#ax = fig.gca(projection="3d")

# Make data.
x_data = np.arange(0, 1, 0.05)
y_data = np.arange(0, 1, 0.05)
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
#z += eps

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

def designmatrix(x,y):

    x2 = x*x
    y2 = y*y
    x3 = x*x*x
    y3 = y*y*y

    X = np.c_[np.ones((len(x),1)),x,x2,y2,x*y,x3,y3,x*y2,x2*y,
                     x*x3,y*y3,x3*y,x*y3,x2*y2,x3*x2,y3*y2,(x2*x2)*y,
                     x*(y2*y2),x3*y2,x2*y3]

    return X



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

k = 5
scores_MSE = np.zeros(k)
scores_R2 = np.zeros(k)

kfold = KFold(n_splits = k)
i=0
for train_inds, test_inds in kfold.split(x):
    xtrain = x[train_inds]
    ytrain = y[train_inds]
    ztrain = z[train_inds]

    xtest = x[test_inds]
    ytest = y[test_inds]
    ztest = z[test_inds]

    #(len(xtrain),len(x), len(xtest))
    Xtrain = designmatrix(xtrain,ytrain)
    betatrain = fitBeta(Xtrain,ztrain)

    Xtest = designmatrix(xtest,ytest)

    zpred = predictor(Xtest,betatrain)
    #print(len(betatrain), len(Xtest))

    scores_R2[i] = R2(ztest, zpred)
    print ("R2",scores_R2[i])
    scores_MSE[i] = MSE(ztest, zpred)
    print("MSE", scores_MSE[i])
    i += 1

estimated_mse_KFold = np.mean(scores_MSE)
estimated_R2_KFold = np.mean(scores_R2)

print("Mean R2",estimated_R2_KFold)
print("Mean MSE",estimated_mse_KFold)
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

# Plot the surface.
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
