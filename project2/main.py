from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import SGDRegressor
from random import random, seed

#import functions
from functions.neuralnetwork import NeuralNetwork
from functions.functions import *
"""
Simple filter for sklearn
"""
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


cancer = load_breast_cancer()
#print(len(cancer.target))

# Set up training data
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)


# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

theta_linreg = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
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
accuracy_score = float(I/len(y_tilde))
print ("-------------------------------------------------")
print ("--- Logistic Regression with Gradient Descent ---")
print ("-------------------------------------------------")
print ("Accuracy score on test set:", accuracy_score)

"""
sgdreg = SGDRegressor(max_iter = m, penalty=None, eta0=eta)
sgdreg.fit(X_train,y_train)
print("beta from sklearn sdg")
#print(sgdreg.intercept_, sgdreg.coef_)
"""

"""
#TEST AGAINST SKLEARN Logistic Regression
#Logistic regression with sklearn
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(logreg.score(X_test,y_test)))
"""

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
print ("---------- Artificial Neural Network ------------")
print ("-------------------------------------------------")
# building our neural network
n_inputs, n_features = X_train.shape
n_hidden_neurons = 100
n_categories = 2


epochs = 100
batch_size = 100
lmbd = 0.


y_train_onehot = to_categorical_numpy(y_train)

dnn = NeuralNetwork(X_data = X_train,Y_data= y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
dnn.train()
test_predict = dnn.predict(X_test)
# equivalent in numpy

print("Accuracy score on test set:", accuracy_score_numpy(y_test, test_predict))



#print("Accuracy score on test set: ", accuracy_score_numpy(Y_test, test_predict))
