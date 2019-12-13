import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import tqdm as tqdm
import os




# Defining analytical eigenvalues
j = np.array([1,2,3,4,5,6])
jpi = j*np.pi
x_analytic = 2-2*tf.cos(jpi/(len(j)+1))

# Defining a 6x6 matrix with only zeros
A = np.zeros((6, 6))



# Setting up the matrix elements
A[0][0] = 2
A[0][1] = -1
for i in range(0,5):
    A[i][i+1] = -1
    A[i][i] = 2
    A[i+1][i] = -1



A[5][5] = 2
A[5][4] = -1


# Changing the sign of matrix A; A gives max lambda and -A gives min lambda
#A = -A

# Defining the 6x6 identity matrix
I = np.identity(6)

dt = np.ones(6)

x0 = np.random.rand(6)



# The construction phase
# Convert the values the trial solution is evaluated at to a tensor.

I_tf = tf.convert_to_tensor(I)
x0_tf = tf.convert_to_tensor(np.random.random_sample(size = (1,6)))
dt_tf = tf.reshape(tf.convert_to_tensor(dt),shape=(-1,1))

num_iter = 30000
num_hidden_neurons = [100] # Number of hidden neurons in each layer

points = tf.concat([x0_tf],1)


with tf.variable_scope('dnn'):
    num_hidden_layers = np.size(num_hidden_neurons)

    #previous_layer = points
    previous_layer = x0_tf

    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
        previous_layer = current_layer

    dnn_output = tf.layers.dense(previous_layer, 6)



def f(x):
    xT = tf.transpose(x)
    xTx = tf.linalg.matmul(xT, x)
    xTxA = xTx*A

    xTA = tf.linalg.matmul(xT, A)
    xTAx = tf.linalg.matmul(xTA, x)

    B = xTxA - (1 - xTAx)*I_tf
    return ( tf.linalg.matmul(B, x) )

def cost_func(x):

    xTxA = (tf.tensordot(tf.transpose(x), x, axes=1)*A)
    # (1- xTAx)*I
    xTAxI = (1- tf.tensordot(tf.transpose(x), tf.tensordot(A, x, axes=1), axes=1))*np.eye(6)
    # (xTx*A - (1- xTAx)*I)*x
    f = tf.tensordot((xTxA + xTAxI), x, axes=1)

    return(f)  # x(t))


with tf.name_scope('loss'):
    # Define the trial solution
    print("dnn_output = ", dnn_output)

    x_trial = tf.transpose(dnn_output)  # x(t)

    print("x_trial = ", x_trial)
    #right_side = f(x_trial) # -x(t) + f(x(t))
    right_side = tf.transpose(cost_func(x_trial))

    print(right_side)

    x_trial = tf.transpose(x_trial)  # x(t)
    # Define the cost function
    loss = tf.losses.mean_squared_error(right_side, x_trial)


learning_rate = 1e-1
with tf.name_scope('train'):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer()

    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

x_dnn = None



losses = []
## The execution phase
with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        sess.run(training_op)

        if i % 100 == 0:
            l = loss.eval()
            print("Step:", i, "/",num_iter, "loss: ", l, "Eigenvalue:" , x_trial.eval() @ (A @ x_trial.eval().T) / (x_trial.eval() @ x_trial.eval().T))

            losses.append(l)



    x_analytic = x_analytic.eval()

    x_dnn = x_trial.eval()
x_dnn = x_dnn.T


eigenvalue = x_dnn.T @ (A @ x_dnn) / (x_dnn.T @ x_dnn)


## Compare with the analytical solution
print("\n Analytical Eigenvalues: \n", x_analytic)
print("\n Final Numerical Eigenvalue \n", eigenvalue)
diff = np.min(abs(x_analytic - eigenvalue))
print("\n")
print('Absolute difference between Analytical Eigenvalue and TensorFlow DNN = ',np.max(diff))
