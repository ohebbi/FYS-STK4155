# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:02:45 2019

@author: mohe9
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
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

x2 = x*x
y2 = y*y
x3 = x*x*x
y3 = y*y*y


DesignMatrix = np.c_[np.ones((len(x),1)),x,x2,y2,x*y,x3,y3,x*y2,x2*y,
                     x*x3,y*y3,x3*y,x*y3,x2*y2,x3*x2,y3*y2,(x2*x2)*y,
                     x*(y2*y2),x3*y2,x2*y3]
X = DesignMatrix

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)


# Now we compute y = Xb
z_tilde = X.dot(beta)

#adding normal distribution
eps = np.random.normal(0,1,len(z_tilde))
z_tilde += eps*0.01 

# We use now Scikit-Learn's linear regressor for control of our results
clf = skl.LinearRegression().fit(X, z)
z_tilde_skl = clf.predict(X)

#print(ztilde-ztilde_skl)

def R2(z_data, z_model):
    return (1 - np.sum((z_data - z_model)**2)/np.sum((z_data - np.mean(z_data))**2))
print (R2(z,z_tilde))

def MSE(z_data,z_model):
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))
print (MSE(z,z_tilde))

from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error for sklearn: %f" % mean_squared_error(z, z_tilde))
print("R2 score for sklearn: %f " % r2_score(z, z_tilde))

from sklearn.model_selection import train_test_split
Z_train, Z_test = train_test_split(z_tilde, test_size=0.33)

print (len(Z_train), len(Z_test))


from sklearn.model_selection import KFold
scores_KFold = np.zeros((nlambdas, k))



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


