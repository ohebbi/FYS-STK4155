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
g_analytic = 2-2*tf.cos(jpi/(len(j)+1))

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

t = np.random.rand(6)



# The construction phase
# Convert the values the trial solution is evaluated at to a tensor.
zeros = tf.convert_to_tensor(np.zeros(t.shape))

I_tf = tf.convert_to_tensor(I)
t_tf = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))
dt_tf = tf.reshape(tf.convert_to_tensor(dt),shape=(-1,1))

num_iter = 1000000
num_hidden_neurons = [100] # Number of hidden neurons in each layer

points = tf.concat([t_tf],1)


with tf.variable_scope('dnn'):
    num_hidden_layers = np.size(num_hidden_neurons)

    previous_layer = points
    #previous_layer = t_tf

    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
        previous_layer = current_layer

    dnn_output = tf.layers.dense(previous_layer, 1)



def f(x):
    xT = tf.transpose(x)
    xTx = tf.linalg.matmul(xT, x)
    xTxA = xTx*A

    xTA = tf.linalg.matmul(xT, A)
    xTAx = tf.linalg.matmul(xTA, x)

    B = xTxA + (1 - xTAx)*I_tf
    return ( tf.linalg.matmul(B, x) )


with tf.name_scope('loss'):
    # Define the trial solution
    g_trial = dnn_output  # x(t)

    # Define the gradient of the trial solution
    g_trial_dt = tf.gradients(g_trial, dnn_output) # dx/dt

    right_side = f(g_trial) - g_trial # -x(t) + f(x(t))

    # Define the cost function
    loss = tf.losses.mean_squared_error(tf.reshape(g_trial_dt,shape=[6,1]), right_side)

    """
    #Using the analytical eigenvalue to calculate the numerical eigenvalue

    vT = tf.transpose(g_trial)
    Av = tf.linalg.matmul(A, g_trial)
    vTAv = tf.linalg.matmul(vT, Av)
    vTv = tf.linalg.matmul(vT, g_trial)
    eig = (vTAv/vTv)

    loss = tf.losses.mean_squared_error(tf.reshape(g_analytic[-1],shape=[1,1]), eig)
    """

learning_rate = 1e-2
with tf.name_scope('train'):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer()

    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

g_dnn = None


best_fit = 100
## The execution phase
with tf.Session() as sess:
    init.run()
    for i in tqdm.tqdm(range(num_iter)):
        sess.run(training_op)

         #If one desires to see how the cost function behaves during training
        #if i % 100 == 0:
            #print(loss.eval())

        if loss.eval() < best_fit:
            g_trial1 = g_trial
            best_fit = loss.eval()
    print("\n MSE: \n", best_fit)

    b = sess.run(g_trial_dt)
    print("\n tf.gradient of g_trial: \n", b)
    g_analytic = g_analytic.eval()

    g_dnn = g_trial1.eval()

    v = g_dnn
    print("\n Numerical eigenvector \n", v)
    vT = tf.transpose(v)
    Av = tf.linalg.matmul(A,v)
    vTAv = tf.linalg.matmul(vT, Av)

    vTv = tf.linalg.matmul(vT, v)

    eig = (vTAv/vTv)
    eigenvalue = eig.eval()

## Compare with the analytical solution
print("\n Analytical eigenvalues: \n", g_analytic)
print("\n Numerical eigenvalue \n", eigenvalue)
diff = np.abs(g_analytic[-1] - eigenvalue)
print("\n")
print('Absolute difference between analytical solution and TensorFlow DNN = ',np.max(diff))

#G_analytic = g_analytic.reshape(6, 1)
#Eigenvalue = eigenvalue.reshape(6, 1)

#diff = np.abs(G_analytic - G_dnn)


"""
# Plot the results

X,T = np.meshgrid(x_np, t_np)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(X,T,G_analytic,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');


# I think the plots for the 2D is wrong, maybe because of wrong slicing below

## Take some 3D slices
indx1 = 0
indx2 = int((1/dt)/2)
indx3 = int(1/dt)-1

t1 = t_np[indx1]
t2 = t_np[indx2]
t3 = t_np[indx3]

# Slice the results from the DNN
res1 = G_dnn[indx1,:]
res2 = G_dnn[indx2,:]
res3 = G_dnn[indx3,:]

# Slice the analytical results
res_analytical1 = G_analytic[indx1,:]
res_analytical2 = G_analytic[indx2,:]
res_analytical3 = G_analytic[indx3,:]

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t1)
plt.plot(x_np, res1, "o")
plt.plot(x_np,res_analytical1)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x_np, res2, "o")
plt.plot(x_np,res_analytical2)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x_np, res3, "o")
plt.plot(x_np,res_analytical3)
plt.legend(['dnn','analytical'])

plt.show()
"""
