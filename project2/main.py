from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from random import random, seed
from pandas import DataFrame

#import functions
import matplotlib.pyplot as plt
from functions.neuralnetwork import NeuralNetwork
from functions.functions import *
#####################
#Simple filter for sklearn
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#####################
#vizualization
import seaborn as sns
sns.set()
#####################

cancer = load_breast_cancer()
#print(len(cancer.target))

# Set up training data
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)


print ("-------------------------------------------------")
print ("--- Logistic Regression with Gradient Descent ---")
print ("-------------------------------------------------")

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#theta_linreg = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
#print("Own inversion")
#print(theta_linreg)


n = len(X_train)
M = 10
m = int(n/M)
n_epochs = 150

t0, t1 = 1, 50

#remember to write about the choice of eta (learning rate).
beta = np.random.randn(len(X_train[0,:]),1)

costfunction_best = 100000
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_train[random_index:random_index+1]
        yi = y_train[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ beta)-yi)
        eta = learning_schedule(epoch*m+i)
        beta = beta - eta*gradients

        Xbeta = X_train.dot(beta)

        costfunction = y_train.dot( Xbeta ) - np.log( 1 + np.exp( Xbeta )  )
        #print(costfunction)
        costfunction = -np.sum(costfunction)

        p = np.exp(Xbeta)/(1+np.exp(Xbeta))

        if (costfunction) < (costfunction_best):
            costfunction_best = costfunction
            best_beta = beta

#print("beta from own sdg")
#print(best_beta, costfunction_best)


y_tilde = X_test.dot(best_beta)

I=0
y_tilde = sigmoid(y_tilde)

for i in range(len(y_tilde)):

    if y_tilde[i]>=0.5:
        y_tilde[i]=1
    else:
        y_tilde[i]=0

    if y_tilde[i] == y_test[i]:
        I += 1.
#print (y_tilde)
score = float(I/len(y_tilde))

print ("Accuracy score on test set:", score)

"""
sgdreg = SGDRegressor(max_iter = m, penalty=None, eta0=eta)
sgdreg.fit(X_train,y_train)
print("beta from sklearn sdg")
#print(sgdreg.intercept_, sgdreg.coef_)
"""


#TEST AGAINST SKLEARN Logistic Regression
#Logistic regression with sklearn
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(logreg.score(X_test,y_test)))


"""
# remove this if scaling of dataset is not required.
# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))
"""
print()
print ("-------------------------------------------------")
print ("-------- classification Neural Network ----------")
print ("-------------------------------------------------")

# building our neural network
n_inputs, n_features = X_train.shape
n_hidden_neurons = 50
n_categories = 2
n_hidden_layers = 2

epochs = 300
batch_size = 100
#lmbd = 0
#eta = 1e-5

y_train_onehot = to_categorical_numpy(y_train)

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

import scikitplot.metrics as skplt
best_data = 0
# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = NeuralNetwork(X_data = X_train, Y_data = y_train_onehot, n_hidden_layers=n_hidden_layers,
                            n_hidden_neurons=n_hidden_neurons, n_categories = n_categories,
                            epochs = epochs, batch_size=batch_size, eta=eta, lmbd=lmbd,
                            user_action = 'classification')
        dnn.train()

        DNN_numpy[i][j] = dnn

        train_pred = dnn.predict(X_train)
        test_pred = dnn.predict(X_test)

        train_accuracy[i][j] = accuracy_score_numpy(y_train, train_pred)
        test_accuracy[i][j] = accuracy_score_numpy(y_test, test_pred)
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", test_accuracy[i][j])
        print()

        #cumulative gain plot
        if test_accuracy[i][j] > best_data:

            best_data = test_accuracy[i][j]
            test_pred_prob = dnn.predict_probabilities(X_test)

x0,y0 = best_curve(y_test,0)
x1,y1 = best_curve(y_test,1)


skplt.plot_cumulative_gain(y_test,test_pred_prob)
plt.plot(x0,y0,'--',color='royalblue')
plt.plot(x1,y1,'--',color='darkorange')
plt.show()
train_accuracy = DataFrame(train_accuracy, index = eta_vals, columns = lmbd_vals)

fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy for classification")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

test_accuracy = DataFrame(test_accuracy, index = eta_vals, columns = lmbd_vals)

fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy for classification")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()


print ("-------------------------------------------------")
print ("-------------Regression Neural Network-----------")
print ("-------------------------------------------------")

x,y,z = generate_data(plott = False) #FrankeFunction

f = z
#scaling with function sigmoid
x = sigmoid(x)
y = sigmoid(y)
z = sigmoid(z)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,random_state=0)

z_train_inv = inv_sigmoid(z_train)
z_test_inv = inv_sigmoid(z_test)


X_train = find_designmatrix(x_train, y_train, polygrad = 5)

n_inputs, n_features = X_train.shape
n_categories = 1
n_hidden_neurons = 50
n_hidden_layers = 2

epochs = 300
batch_size = 100

eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-8, 0, 9)

train_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
test_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))

# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

#Finding designmatrix
X_test = find_designmatrix(x_test, y_test, polygrad = 5)

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = NeuralNetwork(X_data = X_train, Y_data = z_train, n_hidden_layers=n_hidden_layers,
                            n_hidden_neurons=n_hidden_neurons, n_categories = n_categories,
                            epochs = epochs, batch_size=batch_size, eta=eta, lmbd=lmbd,
                            user_action = 'regression')
        dnn.train()

        DNN_numpy[i][j] = dnn

        train_pred = dnn.predict_probabilities(X_train)
        test_pred = dnn.predict_probabilities(X_test)

        train_pred = inv_sigmoid(train_pred)
        test_pred = inv_sigmoid(test_pred)


        train_MSE[i][j] = MSE(z_train_inv, train_pred)[0]
        test_MSE[i][j] = MSE(z_test_inv, test_pred)[0]

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Mean square error on test set:", test_MSE[i][j])
        print()




fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(train_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_title("MSE for regression")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(test_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_title("MSE for regression")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()



g = inv_sigmoid(z)
print(np.mean(f-g))


print ("-------------------------------------------------")
print ("----Sklearn Regression Neural Network-----------")
print ("-------------------------------------------------")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale

x,y,z = generate_data(plott = False) #FrankeFunction

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,random_state=0)

X_train = find_designmatrix(x_train, y_train, polygrad = 5)

X_test = find_designmatrix(x_test, y_test, polygrad = 5)

#Scaling data
X_train = scale(X_train)
z_train = scale(z_train)
X_test = scale(X_test)
z_test = scale(z_test)



dnn = MLPRegressor(hidden_layer_sizes=(100), activation='relu',
                            max_iter = 1000
                            )
dnn.fit(X_train,z_train)

#Printing lambda and eta
print("NN parameters: ")

parameters = dnn.get_params()

print (u'\u03BB =', parameters["alpha"])
print (u"\u03B7 =", parameters["learning_rate_init"])
print("R2 score on test set: ", dnn.score(X_test, z_test))
print()



print ("-------------------------------------------------")
print ("----Sklearn classification Neural Network-----------")
print ("-------------------------------------------------")

import scikitplot.metrics as skplt
from sklearn.neural_network import MLPClassifier

cancer = load_breast_cancer()
# Set up training data
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)


dnn = MLPClassifier(hidden_layer_sizes=(100), activation = 'logistic',
                              max_iter = 1000)
dnn.fit(X_train,y_train)

print("NN parameters: ")

#Printing lambda and eta
parameters = dnn.get_params()

print (u'\u03BB =', parameters["alpha"])
print (u"\u03B7 =", parameters["learning_rate_init"])
print("Accuracy score on test set: ", dnn.score(X_test, y_test))


y_train_prob = dnn.predict_proba(X_train)
y_test_prob = dnn.predict_proba(X_test)



x0,y0 = best_curve(y_train,0)
x1,y1 = best_curve(y_train,1)
skplt.plot_cumulative_gain(y_train,y_train_prob)
plt.title("Cumulative Gains Curve for training data")
plt.plot(x0,y0,'--',color='royalblue')
plt.plot(x1,y1,'--',color='darkorange')
plt.ylim(0,1.05)
plt.savefig("plots/classification/sklearn_training_cumulative.pdf")
plt.show()

x0,y0 = best_curve(y_test,0)
x1,y1 = best_curve(y_test,1)

skplt.plot_cumulative_gain(y_test,y_test_prob)
plt.title("Cumulative Gains Curve for testing data")
plt.ylim(0,1.05)
plt.plot(x0,y0,'--',color='royalblue')
plt.plot(x1,y1,'--',color='darkorange')
plt.savefig("plots/classification/sklearn_test_cumulative.pdf")
plt.show()
