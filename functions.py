import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_data(plott = False):
    """
    Generates data.
    """
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
    z += 0.*eps

    return x, y, z
    #print (x_train)

def terrain_data(plott = True):
    """

    """

    #load the terrain
    terrain1 = imread('terraintiff.tif')

    #Reducing the size of the terrain data to improve computation time
    z_data = terrain1[::30,::30]
    x_data = np.arange(0,len(z_data[0]),1)
    y_data = np.arange(0,len(z_data[:,0]),1)
    x, y = np.meshgrid(x_data,y_data)

    if plott == True:
        fig1 = plt.figure()
        plt.title("Terrain over a part of Norway")
        image = plt.imshow(z_data)
        plt.colorbar(image)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig("Aktuelt område.png")
        plt.show()

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z_data)

    return x, y, z

def find_designmatrix(x,y, polygrad=5):

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

    #General formula to avoid hardcoding too much.

    elif polygrad > 5:
        X = np.zeros( (len(x), int(0.5*(polygrad + 2)*(polygrad + 1)) ) )
        poly = 0
        for i in range(int(polygrad) + 1):
            for j in range(int(polygrad) + 1 - i):
                X[:,poly] = (x**i)*(y**j)
                poly += 1
    return X

def R2(z_data, z_model):
    return (1 - np.sum( (z_data - z_model)**2 ) / np.sum((z_data - np.mean(z_data))**2))

def MSE(z_data,z_model):
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))

#We are approximating variance to be equal MSE
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

def lasso_regression(X,z,lamb):
    clf = Lasso(alpha=lamb, tol = 1)
    clf.fit(X,z)
    return (clf.coef_)
def crossvalidation(x_train, y_train, z_train, x_test, y_test, z_test, k, polygrad, regressiontype = 'OLS',lamb=0):

    scores_MSE = np.zeros(k)
    scores_R2 = np.zeros(k)

    #finding correct length of beta with the beautiful dummy variabe
    dummy_variable = find_designmatrix(np.zeros(1),np.zeros(1),polygrad)
    beta_perfect = np.zeros(len(dummy_variable[0]))

    print
    kfold = KFold(n_splits = k, shuffle=True)

    z_ALL_pred = np.empty((z_test.shape[0], k))
    #splitting our training data into training- and validation data
    i =0
    for train_inds, test_inds in (kfold.split(x_train)):
        xtrain = x_train[train_inds]
        ytrain = y_train[train_inds]
        ztrain = z_train[train_inds]

        xtest2 = x_train[test_inds]
        ytest2 = y_train[test_inds]
        ztest2 = z_train[test_inds]
        #(len(xtrain),len(x), len(xtest))
        Xtrain = find_designmatrix(xtrain,ytrain, polygrad)

        if regressiontype == 'OLS':
            betatrain = OLS(Xtrain,ztrain)
        elif regressiontype == 'Ridge':
            betatrain = ridge_regression(Xtrain, ztrain, lamb)
        elif regressiontype == 'Lasso':
            betatrain = lasso_regression(Xtrain, ztrain, lamb)
        else:
            raise ValueError ("regression-type is lacking input!")
        #print (i)
        Xtest = find_designmatrix(x_test,y_test,polygrad)

        z_ALL_pred[:, i] = Xtest @ betatrain

        #print ( z_ALL_pred[:,i])
        #print(len(betatrain), len(Xtest))


        #scores_MSE[i] =  MSE(ztest,z_ALL_pred[:, i])
        #scores_R2[i] = R2(ztest,z_ALL_pred[:,i])

        #print (len(beta_perfect), len(betatrain))
        beta_perfect += betatrain
        i += 1

    print (z_ALL_pred)
    print (np.mean(z_ALL_pred,axis=1,keepdims=True))


    error = np.mean( np.mean((z_test[:,np.newaxis] - z_ALL_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(z_ALL_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_ALL_pred, axis=1, keepdims=True) )


    #print('Polynomial degree:', degree)
    """
    print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))
    """
    #estimated_MSE = np.mean(scores_MSE)

    return (estimated_MSE, bias, variance, error, (beta_perfect/float(k)))



def k_fold_cross_validation(x, y, z, polygrad, k=5, lamb=0, regressiontype = 'OLS'):

    p = int(0.5*(polygrad + 2)*(polygrad + 1))

    scores_MSE = np.zeros(k)

    bias = np.zeros(k)
    variance = np.zeros(k)
    scores_R2 = np.zeros(k)
    betas = np.zeros((p,k))

    #finding correct length of beta with the beautiful dummy variabe
    """
    dummy_variable = find_designmatrix(np.zeros(1),np.zeros(1),polygrad)
    beta_perfect = np.zeros(len(dummy_variable[0]))
    """

    kfold = KFold(n_splits = k, shuffle=True)

    #splitting our training data into training- and validation data
    i =0
    for train_inds, val_inds in kfold.split(x):
        xtrain = x[train_inds]
        ytrain = y[train_inds]
        ztrain = z[train_inds]

        xval = x[val_inds]
        yval = y[val_inds]
        zval = z[val_inds]

        #(len(xtrain),len(x), len(xtest))
        Xtrain = find_designmatrix(xtrain,ytrain, polygrad)

        if regressiontype == 'OLS':
            betatrain = OLS(Xtrain,ztrain)
        elif regressiontype == 'Ridge':
            betatrain = ridge_regression(Xtrain, ztrain, lamb)
        elif regressiontype == 'Lasso':
            betatrain = lasso_regression(Xtrain, ztrain, lamb)
        else:
            raise ValueError ("regression-type is lacking input!")

        Xval = find_designmatrix(xval,yval,polygrad)
        zpred = Xval @ betatrain

        scores_MSE[i] =  MSE(zval,zpred)
        scores_R2[i] = R2(zval,zpred)
        #print (len(beta_perfect), len(betatrain))
        betas[:,i] = betatrain
        i += 1

    estimated_MSE = np.mean(scores_MSE)
    estimated_R2 = np.mean(scores_R2)
    return [estimated_MSE, estimated_R2], betas

def bootstrap(x,y,z,degrees,regressiontype):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.utils import resample
    n_bootstrap = 100
    maxdegree = int(degrees[-1])
    print(maxdegree)
    error_test = np.zeros(maxdegree)
    error_train = np.zeros(maxdegree)
    bias =  np.zeros(maxdegree)
    variance =  np.zeros(maxdegree)
    polydegree =  np.zeros(maxdegree)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,test_size  = 0.2)

    for degree in degrees:

        z_ALL_pred = np.empty((z_test.shape[0],n_bootstrap))

        for i in range(n_bootstrap):

            x_,y_,z_ = resample(x_train,y_train,z_train)

            Xtrain = find_designmatrix(x_,y_, degree)

            if regressiontype == 'OLS':
                betatrain = OLS(Xtrain,z_)
            elif regressiontype == 'Ridge':
                betatrain = ridge_regression(Xtrain, z_)
            elif regressiontype == 'Lasso':
                betatrain = lasso_regression(Xtrain, z_)
            else:
                raise ValueError ("regression-type is lacking input!")

            Xtest = find_designmatrix(x_test,y_test,degree)

            z_ALL_pred[:, i] = (Xtest @ betatrain).ravel()

        #print (z_train)
        z_test = np.reshape(z_test,(len(z_test),1))

        error_test[int(degree)-1] = np.mean( np.mean( ( z_test - z_ALL_pred)**2,axis=1,keepdims=True) )

    return error_test

def bias_variance(x, y, z, polygrad, k, lamb=0, regressiontype = 'OLS'):

    x, x_test, y, y_test, z, z_test = train_test_split(x,y,z,test_size=0.2)

    scores, betas = k_fold_cross_validation(x, y, z, polygrad, k, regressiontype)
    MSE_train = scores[0]
    R2_train = scores[1]

    X_test = find_designmatrix(x_test, y_test, polygrad=polygrad)
    z_pred = X_test @ betas

    z_test = np.reshape(z_test,(len(z_test),1))

    MSE_test = np.mean( np.mean(( z_test - z_pred)**2,axis=1,keepdims=True) )
    bias_test = np.mean( (z_test - np.mean(z_pred, axis = 1, keepdims=True))**2)
    variance_test = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    return [MSE_train,R2_train, MSE_test, bias_test, variance_test], betas


def Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='OLS'):

    """
    First running CV for finding best lambda with lowest MSE.
    """


    test_MSE = np.zeros(len(degrees))
    test_R2 = np.zeros(len(degrees))

    for polygrad in degrees:

        j = int(polygrad) - 1

        scores, beta = k_fold_cross_validation(x, y, z, polygrad, k, lamb, regressiontype)

        test_MSE[j] = scores[0]
        test_R2[j] = scores[1]

    return test_MSE


def Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='OLS'):

    """
    Then calculate the bias-variance with the best lambda.
    As an example we now use "best" lambda = 0.1
    """
    train_MSE = np.zeros(len(degrees))
    train_R2 = np.zeros(len(degrees))

    test_MSE = np.zeros(len(degrees))
    test_R2 = np.zeros(len(degrees))
    bias = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    for polygrad in degrees:

        j = int(polygrad) - 1

        scores, betas = bias_variance(x, y, z, polygrad, k, lamb, regressiontype)

        train_MSE[j] = scores[0]
        train_R2[j] = scores[1]

        test_MSE[j] = scores[2]
        bias[j] = scores[3]
        variance[j] = scores[4]

    return test_MSE, bias, variance, betas
