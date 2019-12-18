import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d
import tqdm as tqdm
import os
import seaborn as sns



def NN_Eigenvalue(matrix_size, string1, run_iter, num_iter, num_hidden_neurons, string2, learning_rate):

    # Defining values for nice plotting
    color=iter(cm.rainbow(np.linspace(1,0,3)))
    plt.rcParams["font.family"]= "Times New Roman"
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    #fig, ax = plt.subplots()
    sns.set()


    # Defining analytical eigenvalues
    j = np.zeros(matrix_size)
    for i in range(matrix_size):
        j[i] = i+1
    jpi = j*np.pi
    analytic_eigenvalue = 2-2*np.cos(jpi/(len(j)+1))

    # Defining a 6x6 matrix with only zeros
    C = np.zeros((matrix_size, matrix_size))

    # Setting up the matrix elements
    for i in range(matrix_size-1):
        C[i][i+1] = -1
        C[i][i] = 2
        C[i+1][i] = -1
    C[matrix_size-1][matrix_size-1] = 2


    B = np.random.random_sample((matrix_size,matrix_size))
    B = (B.T + B)/2.

    if string1 == "analytical":
        A = C
    if string1 == "random":
        A = B


    # Defining the 6x6 identity matrix
    I = np.identity(matrix_size)
    x0 = np.random.rand(matrix_size)
    x0 = x0/np.sqrt(np.sum(x0*x0))


    # The construction phase
    # Convert the values the trial solution is evaluated at to a tensor.
    I_tf = tf.convert_to_tensor(I)
    x0_tf = tf.convert_to_tensor(np.random.random_sample(size = (1,matrix_size)))

    lambdas = np.zeros((run_iter,num_iter))

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = x0_tf

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, matrix_size)



    def cost_func(x):

        xTxA = (tf.tensordot(tf.transpose(x), x, axes=1)*A)
        # (1- xTAx)*I
        xTAxI = (1- tf.tensordot(tf.transpose(x), tf.tensordot(A, x, axes=1), axes=1))*np.eye(matrix_size)
        # (xTx*A - (1- xTAx)*I)*x
        f = tf.tensordot((xTxA + xTAxI), x, axes=1)

        return(f)  # x(t))

    for runs in range(run_iter):
        with tf.name_scope('loss'):
            # Define the trial solution
            x_trial = tf.transpose(dnn_output)  # x(t)

            right_side = tf.transpose(cost_func(x_trial))

            x_trial = tf.transpose(x_trial)  # x(t)
            # Define the cost function
            loss = tf.losses.mean_squared_error(right_side, x_trial)


        with tf.name_scope('train'):
            if string2 == "GD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            if string2 == "Adam":
                optimizer = tf.train.AdamOptimizer()

            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        x_dnn = None

        num_iter_needed = []
        x_trial_list = np.zeros((num_iter,matrix_size))

        ## The execution phase
        with tf.Session() as sess:
            init.run()
            for i in range(num_iter):
                sess.run(training_op)

                eigenvalue = (x_trial.eval() @ (A @ x_trial.eval().T)
                            /(x_trial.eval() @ x_trial.eval().T))[0,0]

                lambdas[runs][i] = eigenvalue

                x_dnn = x_trial.eval()
                x_dnn = x_dnn.T

                ## Normalize g_trials for comparing to normalized numpy solution
                x_dnn = x_dnn/np.sqrt(np.sum(x_dnn*x_dnn))

                if i % 100 == 0:
                    l = loss.eval()
                    print("Step:", i, "/",num_iter, "loss: ", l, "Eigenvalue:" , eigenvalue)
                    if l==0:#<1e-16:

                        x_trial_list[i:]=x_trial_list[i-1]
                        lambdas[runs][i:]=eigenvalue
                        num_iter_needed.append(i)
                        break


                for l in range(matrix_size):
                    x_trial_list[i][l] =  x_dnn[l]

            # Finding eigenvector using Numpy
            numpy_eigenvalue, numpy_vector = np.linalg.eig(A)

            # Visualization
            iterations=np.linspace(1,num_iter,num_iter)

    c = next(color)
    for k in range(matrix_size):
        tmp_xaxis = len(x_trial_list[:,k])
        if k==0:
            plt.plot(np.linspace(1,tmp_xaxis,tmp_xaxis),x_trial_list[:,k],c=c,label="%2.5f" % eigenvalue,linewidth=0.9)
        else:
            plt.plot(np.linspace(1,tmp_xaxis,tmp_xaxis),x_trial_list[:,k],c=c,linewidth=0.9)

    if string1 == "analytical":
        ## Compare with the analytical solution
        print("\n Analytical Eigenvalues: \n", analytic_eigenvalue)
        print("\n Final Numerical Eigenvalue \n", eigenvalue)
        diff = np.min(abs(analytic_eigenvalue - eigenvalue))
        print("\n")
        print('Absolute difference between Analytical Eigenvalue and TensorFlow DNN = ',diff)

        if string1 == "random":
            ## Compare with the analytical solution
            print("\n Numpy Eigenvalues: \n", numpy_eigenvalue)
            print("\n Final Numerical Eigenvalue \n", eigenvalue)
            diff = np.min(abs(numpy_eigenvalue - eigenvalue))
            print("\n")
            print('Absolute difference between Analytical Eigenvalue and TensorFlow DNN = ',diff)

    # Finding which element of analytic our estimated eigenvalue converges to

    index = np.argmin(abs(numpy_eigenvalue - eigenvalue))

    #print(numpy_vector)
    #print(x_dnn.T)
    c = next(color)
    for l in range(matrix_size):
        plt.hlines(numpy_vector[index][l],0,num_iter,colors=c,linestyles="dashed",linewidth=2)
        if l==matrix_size-1:
            plt.hlines(-numpy_vector[index][l],0,num_iter,colors=c,linestyles="dashed",label="%2.5f" % numpy_eigenvalue[index],linewidth=2)
            break
        plt.hlines(-numpy_vector[index][l],0,num_iter,colors=c,linestyles="dashed",linewidth=2)
    plt.xlabel(r"Number of iterations", size=12)
    plt.ylabel(r"Value of the elements of the estimated eigenvector",size=12)
    plt.legend()
    plt.title(r"Convergence of the estimated eigenvector",size=12)
    plt.savefig("convergence_eigenvector.pdf")
    plt.show()

    color=iter(cm.rainbow(np.linspace(1,0,run_iter+1)))

    for i in range(runs+1):
        plt.plot(iterations,lambdas[i],c=c)
        c=next(color)

    for l in range(matrix_size):
        plt.hlines(numpy_eigenvalue[l],0,num_iter,colors=c,linestyles="dashed",label="%2.5f" % numpy_eigenvalue[l],linewidth=2,)
    plt.legend()
    plt.xlabel(r"Number of iterations", size=12)
    plt.ylabel(r"Value of the estimated eigenvalue",size=12)
    plt.title(r"Convergence of the estimated eigenvalue",size=12)
    plt.savefig("convergence_eigenvalue.pdf")
    plt.show()

    #sns.set(color_codes=True)
    if run_iter > 1:
        sns.distplot(np.reshape(lambdas[:,-1],[1,int(run_iter)]), bins=matrix_size*2, kde=False, rug=True)
        plt.ylabel(r"Counts of each eigenvalue",size=12)
        plt.xlabel(r"Eigenvalues", size=12)

        #plt.xlabel(r"Number of iterations", size=12)
        plt.savefig("histogram.pdf")
        plt.show()

    print("Mean of number of iterations needed:", np.mean(num_iter_needed))
