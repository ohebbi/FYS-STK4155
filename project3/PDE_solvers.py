# Code for solving the 1+1 dimensional diffusion equation
# du/dt = ddu/ddx on a rectangular grid of size L x (T*dt),
# with with L = 1, u(x,0) = g(x), u(0,t) = u(L,t) = 0

import numpy, sys, math
import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import tqdm as tqdm

# The Neural Network doesn't use Forward Euler
# Therefore it doens't need to follow the stability criteria
def NN_PDE(dx, num_iter, num_hidden_neurons, string, learning_rate):

    # Number of integration points along x-axis
    N       =   int(1.0/dx)

    # Step length in time
    dt      =   dx
    # Number of time steps till final time
    T       =   int(1.0/dt)

    L = 1
    x_np = np.linspace (0,1,N+2)
    t_np = np.linspace(0,1,T)

    x_mesh,t_mesh = np.meshgrid(x_np, t_np)

    x = x_mesh.ravel()
    t = t_mesh.ravel()

    ## The construction phase

    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
    x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
    t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

    points = tf.concat([x,t],1)


    x_mesh_tf = tf.convert_to_tensor(x_mesh)
    t_mesh_tf = tf.convert_to_tensor(t_mesh)


    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)


    ## Define the trial solution and cost function
    def u(x):
        return tf.sin(np.pi*x)


    with tf.name_scope('loss'):
        g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output

        g_trial_dt =  tf.gradients(g_trial,t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)

        loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])


    with tf.name_scope('train'):
        if string == "GD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        if string == "Adam":
            optimizer = tf.train.AdamOptimizer()
        traning_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_analytic = tf.exp(-np.pi**2*t)*tf.sin(np.pi*x)
    g_dnn = None

    ## The execution phase
    with tf.Session() as sess:
        init.run()
        for i in tqdm.tqdm(range(num_iter)):
            sess.run(traning_op)

            # If one desires to see how the cost function behaves during training
            #if i % 100 == 0:
            #    print(loss.eval())

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()


    ## Compare with the analytical solution
    diff = np.abs(g_analytic - g_dnn)
    #print('Max absolute difference between analytical solution and TensorFlow DNN = ',np.max(diff))

    G_analytic = g_analytic.reshape((int(1.0/dt),int(1.0/dx)+2))
    G_dnn = g_dnn.reshape((int(1.0/dt),int(1.0/dx)+2))

    diff = np.abs(G_analytic - G_dnn)
    return T, x_mesh, t_mesh, G_dnn


def forward_step(alpha, u, uPrev, N, string):
    """
    Steps forward-euler algo one step ahead.
    Implemented in a separate function for code-reuse from crank_nicolson()
    """
    if string == "FE":
        for x in range(1,N+1): #loop from i=1 to i=N
            u[x] = alpha*uPrev[x-1] + (1-2*alpha)*uPrev[x] + alpha*uPrev[x+1]

    if string == "CN":
        for x in range(1,N+1): #loop from i=1 to i=N
            u[x] = alpha*uPrev[x-1] + (2-2*alpha)*uPrev[x] + alpha*uPrev[x+1]

def forward_euler(alpha,u,N,T):
    """
    Implements the forward Euler sheme, results saved to
    array u
    """
    #Skip boundary elements
    for t in range(1,T):
        forward_step(alpha, u[t], u[t-1], N, "FE")

def tridiag(u, u_prev, alpha, N, string):
    """
    Tridiagonal gaus-eliminator, specialized to diagonal = 1+2*alpha,
    super- and sub- diagonal = - alpha
    """
    if string == "BE":
        diag = 1+2*alpha

    if string == "CN":
        diag = 2+2*alpha

    offdiag = -alpha
    d = numpy.zeros(N+1)
    d[0] = diag

    u_old = numpy.zeros(N+1)
    u_old[0] = u_prev[0]

     # Forward substitution
    for i in range(1,N+1):
        btemp = offdiag/d[i-1]
        d[i] = diag - offdiag*btemp
        u_old[i] = u_prev[i] - u_old[i-1]*btemp

    # Special case, boundary conditions
    #u[0] = 0
    #u[N+1] = 0

 #Backward substitute
    for i in range(N, 1, -1):
       u[i] = (u_old[i] - offdiag*u[i+1])/d[i]


def backward_euler(alpha,u,N,T):
    """
    Implements backward euler scheme by gaus-elimination of tridiagonal matrix.
    Results are saved to u.
    """
    for t in range(1,T):
        tridiag(u[t], u[t-1], alpha, N, "BE") #Note: Passing a pointer to row t, which is modified in-place

def crank_nicolson(alpha,u,N,T):
    """
    Implents crank-nicolson scheme, reusing code from forward- and backward euler
    """
    for t in range(1,T):
        u_temp = u.copy()
        forward_step(alpha, u_temp[t], u_temp[t-1], N, "CN")
        tridiag(u[t], u_temp[t], alpha, N, "CN")


def g(x):
    """Initial condition u(x,0) = g(x), x \in [0,1]"""
    return numpy.sin(math.pi*x)
