import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, train_test_split
import sklearn.linear_model as skl
import tqdm as tqdm

#For bootstrap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

def FrankeFunction(x,y):
    """
    Generates Franke's function.
    Input:
    Takes array x and y.
    Output
    Returns array z.
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def generate_data(number_points = 20, plott = True):
    """
    Generates data.
    Input:
    plott = True for plotting.
    Output:
    returns 1D arrays x, y and z (after begin raveled).
    """
    x_data = np.arange(0, 1, 1./number_points)
    y_data = np.arange(0, 1, 1./number_points)

    x, y = np.meshgrid(x_data,y_data)

    z = FrankeFunction(x, y)
    if plott == True:
        plotter(x,y,z)
        plt.savefig('plots/Franke/frankefunction.pdf')

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    print ("x ranges from", 0, "to", 1, "with a total amount of", number_points, "points.")
    print ("y ranges from", 0, "to", 1, "with a total amount of", number_points, "points.")


    eps = np.random.normal(0,1,len(z))
    z += 0.1*eps

    return x, y, z
    #print (x_train)
def plotter(x,y,z):
    """
    Function:
    Generates a three dimensional plot.
    Input:
    Takes an array x, y and z.
    Output:
    Gives a plot.
    """
    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    for angle in range(0,150):
        ax.view_init(40,angle)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5);

def terrain_data(skip_nr_points=50 ,plott = True):
    """
    Generates the terrain data.
    Input:
    Skip number of points (lower number to include more data but will be slow)
    plott = True for plotting.
    Output:
    returns 1D arrays x, y and z (after begin raveled).
    """

    #load the terrain
    terrain1 = imread('datafiles/terraintiff.tif')

    #Reducing the size of the terrain data to improve computation time
    z_data = terrain1[::skip_nr_points,::skip_nr_points]

    x_data = np.linspace(0,1,len(z_data[0]))
    y_data = np.linspace(0,1,len(z_data[:,0]))

    x, y = np.meshgrid(x_data,y_data)


    z = z_data
    z = (z - np.mean(z))/np.sqrt(np.var(z))

    if plott == True:

        fig = plt.figure();
        ax = fig.gca(projection='3d');
        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False);
                           # Customize the z axis.
        #ax.set_zlim(-0.10, 1.40);
        ax.zaxis.set_major_locator(LinearLocator(10));
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
        for angle in range(0,150):
            ax.view_init(60,angle)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5);
        plt.title("Terrain Data")
        plt.savefig("plots/Terrain/3D_plot_TERRAIN.pdf")

        fig1 = plt.figure()
        plt.title("Terrain over a part of Norway")
        image = plt.imshow(z_data)
        plt.colorbar(image)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig("plots/Terrain/2D_plot_TERRAIN.pdf")
        plt.show()

    #flatten the matrix out
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z_data)

    print ("x ranges from", 0, "to", 1, "with a total amount of", len(x), "points.")
    print ("y ranges from", 0, "to", 1, "with a total amount of", len(y), "points.")

    z = (z - np.mean(z))/np.sqrt(np.var(z))

    return x, y, z

def find_designmatrix(x,y, polygrad=5):
    """
    Function:
    Generates the designmatrix.
    Input:
    Takes an array x and y and a polynomial degree.
    Output:
    Returns a multidimensional array (designmatrix).
    """
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

    #General formula to avoid hardcoding 'too' much.
    elif (polygrad > 5):
        X = np.zeros( (len(x), int(0.5*(polygrad + 2)*(polygrad + 1)) ) )
        poly = 0
        for i in range(int(polygrad) + 1):
            for j in range(int(polygrad) + 1 - i):
                X[:,poly] = np.squeeze((x**i)*(y**j))
                poly += 1
    return X

def R2(z_data, z_model):
    """
    Function:
    Finds the R2-level for a given model and approximation.
    Input:
    Takes an array z_data and z_model.
    Output:
    Returns a scalar.
    """
    return (1 - np.sum( (z_data - z_model)**2 ) / np.sum((z_data - np.mean(z_data))**2))

def MSE(z_data,z_model):
    """
    Function:
    Finds the mean square error for a given model and approximation.
    Input:
    Takes an array z_data and z_model.
    Output:
    Returns a scalar.
    """
    summ = 0
    for i in range(len(z_data)):
        summ += (z_data[i] - z_model[i])**2
    return summ/(len(z_data))

def confidence_interval(beta, MSE):
    """
    Function:
    Finds the confidence interval.
    We are approximating the variance to be equal the mean square error.
    Input:
    Array beta and scalar MSE.
    Output:
    Array of three indexes with low, mid, and high beta values.
    """

    sigma = np.sqrt(MSE)
    beta_low = np.mean(beta)-sigma*1.96
    beta_high = np.mean(beta)+sigma*1.96
    return [beta_low, np.mean(beta), beta_high]
def SVDinv(A):
    """
    Function:
    This function inverts matrixes using singular value dcomposition (SVD).

    Input:
    Takes a matrix A.

    Output:
    Returns a matrix.
    """
    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

def OLS(X,z,inversion='SVD'):
    """
    Function:
    This is a solver for the ordinary least square method. Choose 'SVD' for
    numerical stability or choose 'normal' inversion for faster computation.

    Input:
    Takes a design matrix as X, a target-vector as z and inversion type.

    Output:
    Returns the solution array beta of the ordinary least square method.
    """
    if inversion=='normal':
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    elif inversion=='SVD':
        A = X.T.dot(X)
        C = SVDinv(A)
        beta = C.dot(X.T).dot(z)
    return beta

def ridge_regression(X,z,lamb,inversion='SVD'):
    """
    Function:
    This is a solver using Ridge regression. Choose SVD for
    numerical stability or choose normal inversion for faster computation.

    Input:
    Takes a design matrix as X, a target-vector as z, a hyperparameter
    (constant) lambda and a inversion type.


    Output:
    Returns the array solution beta of the Ridge regression.
    """
    if inversion == 'normal':
        beta = np.linalg.inv(X.T.dot(X) + lamb*np.identity(len(X.T.dot(X)))).dot(X.T).dot(z)
    elif inversion == 'SVD':
        A = (X.T.dot(X) + lamb*np.identity(len(X.T.dot(X))))
        C = SVDinv(A)
        beta = C.dot(X.T).dot(z)
    return beta

def lasso_regression(X,z,lamb):
    """
    Function:
    This is a solver using LASSO regression using the module sklearn.

    Input:
    Takes a design matrix as X, a target-vector as z and a hyperparameter
    (constant) lambda.

    Output:
    Returns the array solution beta of the LASSO regression.
    """
    clf = Lasso(alpha=lamb)
    clf.fit(X,z)
    return (clf.coef_)

def k_fold_cross_validation(x, y, z, polygrad, k=5, lamb=0, regressiontype = 'OLS', get_CI = False):
    """
    Function:
    This is a resample-technique based on the k-fold cross-validation.

    Input:
    Takes array input x,y and z as datapoints, the polynomial degree polygrad,
    number of k-fold cross-validation,hyperparameter lamb, a regressiontype,
    and confidence interval of beta.

    Output:
    Returns an array of mse and an array of R2-scores for the training data,
    a matrix with beta values for each k-fold, and a vector with mean beta-
    values.
    """
    p = int(0.5*(polygrad + 2)*(polygrad + 1))
    train_MSE = np.zeros(k)
    betas = np.zeros((p,k))

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

        #training data
        z_train = Xtrain @ betatrain

        train_MSE[i] =  MSE(ztrain,z_train)

        # Storing all the betas
        betas[:,i] = betatrain

        i += 1

    train_MSE = np.mean(train_MSE)
    return [train_MSE], betas

def bootstrap(x,y,z,degrees,lamb=0,regressiontype='OLS',n_bootstrap=100):
    """
    Function:
    This is a resample-technique based on the bootstrap method.

    Input:
    Takes array input x,y and z as datapoints, the polynomial degree interval of
    array degrees, a regressiontype and the number of bootstraps.

    Output:
    Returns an array of mean square error, the bias, and the variance of the
    test data.
    """
    maxdegree = int(degrees[-1])

    error_test =  np.zeros(maxdegree)
    mse        =  np.zeros(maxdegree)
    bias       =  np.zeros(maxdegree)
    variance   =  np.zeros(maxdegree)
    polydegree =  np.zeros(maxdegree)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,test_size  = 0.2)

    for degree in tqdm.tqdm(degrees):

        z_ALL_pred = np.empty((z_test.shape[0],n_bootstrap))
        X_test = find_designmatrix(x_test, y_test, polygrad=degree)


        for i in range(n_bootstrap):

            x_,y_,z_ = resample(x_train,y_train,z_train)

            Xtrain = find_designmatrix(x_,y_, degree)

            if regressiontype == 'OLS':
                betatrain = OLS(Xtrain,z_)
            elif regressiontype == 'Ridge':
                betatrain = ridge_regression(Xtrain, z_, lamb)
            elif regressiontype == 'Lasso':
                betatrain = lasso_regression(Xtrain, z_, lamb)
            else:
                raise ValueError ("regression-type is lacking input!")


            z_ALL_pred[:, i] = (X_test @ betatrain).ravel()

        z_test = np.reshape(z_test,(len(z_test),1))

        error_test[int(degree)-1] = np.mean( np.mean( ( z_test - z_ALL_pred)**2,axis=1,keepdims=True) )
        bias[int(degree)-1]       = np.mean( (z_test - np.mean(z_ALL_pred, axis=1, keepdims=True))**2 )
        variance[int(degree)-1]   = np.mean( np.var(z_ALL_pred, axis=1, keepdims=True) )
    return [error_test, bias, variance]

def bias_variance(x, y, z, polygrad, k, lamb=0, regressiontype = 'OLS'):
    """
    Function:
    Finds the bias and variance for some independent test data.
    Input:
    Takes an array x,y, and z, with polynomial degree polygrad, k number of K-fold
    cross validation, hyperparameter lambda and a regression type.
    Output:
    Returns a list with an array of MSE for train data, and an array of MSE,
    an array of bias and an array of variance for test data, an an array of
    confidence intervals for beta values, and the average beta_values for k-fold CV.
    """
    x, x_test, y, y_test, z, z_test = train_test_split(x,y,z,test_size=0.2)

    scores, beta_k_fold = k_fold_cross_validation(x, y, z, polygrad, k, regressiontype, get_CI = True)
    MSE_train = scores[0]

    X_test = find_designmatrix(x_test, y_test, polygrad=polygrad)
    z_pred = X_test @ beta_k_fold

    z_test = np.reshape(z_test,(len(z_test),1))

    #Calculating different value.
    MSE_test      = np.mean( np.mean(( z_test - z_pred)**2,axis=1,keepdims=True) )
    bias_test     = np.mean( (z_test - np.mean(z_pred, axis = 1, keepdims=True))**2)
    variance_test = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    R2_test       = R2(z_test,np.mean(z_pred,axis=1,keepdims=True))

    CI = confidence_interval(np.mean(beta_k_fold,axis=1,keepdims=True), MSE_test)

    return [MSE_train,R2_test, MSE_test, bias_test, variance_test, CI], beta_k_fold


def Different_Lambdas(x, y, z, degrees, k, lamb, regressiontype='OLS', resample_method='K_fold_CV'):

    """
    Function:
    Runs bias_variance function for different polynomial degrees.
    Input:
    Data x,y and z with an array of polynomial degree degrees, number of k-fold
    cross validation, hyperparameter lamb, and a regression type.
    Output:
    Returns an array with mean square errors for the test data.
    """

    test_MSE = np.zeros(len(degrees))
    test_R2 = np.zeros(len(degrees))

    for polygrad in degrees:

        j = int(polygrad) - 1

        scores, beta_tmp = bias_variance(x, y, z, polygrad, k, lamb, regressiontype)

        test_MSE[j] = scores[2]

    return test_MSE


def Best_Lambda(x, y, z, degrees, k, lamb, regressiontype='OLS'):

    """
    Function:
    Runs bias_variance for different polynomial degrees.
    Input:
    Takes arrays of data x,y and z, with array of polynomial degrees, k number of
    k-fold cross validation, a hyperparameter lamb and regressiontype.
    Output:
    Returns arrays of mean square errors for MSE, R2, bias, variance, confidence
    interval for test data. Then returns a dictionary with beta values for different
    polynomial degrees, and finally returns the mean square error for training
    data.
    """
    train_MSE = np.zeros(len(degrees))
    test_MSE  = np.zeros(len(degrees))

    test_R2  = np.zeros(len(degrees))
    bias     = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    CI  = np.zeros((len(degrees),3))
    bet = {}

    for polygrad in degrees:

        j = int(polygrad) - 1

        scores, bet[int(polygrad)] = bias_variance(x, y, z, polygrad, k, lamb, regressiontype)

        train_MSE[j] = scores[0]
        test_R2[j]   = scores[1]
        test_MSE[j]  = scores[2]
        bias[j]      = scores[3]
        variance[j]  = scores[4]
        CI[j]        = scores[5]

    return test_MSE,test_R2, bias, variance, CI, bet,train_MSE
