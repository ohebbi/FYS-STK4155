import numpy, sys, math
from  matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import cm
import numpy as np
import os
from PDE_solvers import *
from NN_Eigenvalue_solver import *

print("\n Project 3: Neural Network vs PDE's & Eigenvalue Problems")
print(" Which Project Task do you want to run?: ")
print(" Project Task 1 - Solving Partial Differential Equations; Write '1'")
print(" Project Task 2 - Solving Eigenvalue Problems; Write '2'")

Task = input("\n Write here: ")

if Task == "1":
    """
    -------------
    Solving Partial Differential Equations:
    -------------
    """



    print("Which step size of Delta x do you want to run")
    print("dx = 0.1: Write: '0.1' ")
    print("dx = 0.01: Write: '0.01' ")

    # Step length in x
    dx      =   float(input())

    # Number of integration points along x-axis
    N       =   int(1.0/dx)

    # Step length in time
    dt      =   0.5*dx*dx
    # Number of time steps till final time
    T       =   int(1.0/dt)

    L = 1
    x = numpy.linspace (0,1,N+2)
    t = np.linspace(0,1,T)
    alpha = dt/(dx**2)


    # Define method to use 1 = explicit scheme, 2= implicit scheme, 3 = Crank-Nicolson
    print("\n The PDE solvers executing are:")
    print("Forward Euler - Explicit Scheme")
    print("Backward Euler - Implicitt Scheme")
    print("Crank-Nicolson - Implicitt Scheme")



    """
    -------------
    Forward Euler - Explicit Scheme:
    -------------
    """
    uf = np.zeros((t.size,x.size))


    #Initial condition
    uf[0,:] = g(x)
    uf[0,0] = uf[0,N+1] = 0.0 #Implement boundaries rigidly

    forward_euler(alpha,uf,N,T)



    """
    -------------
    Backward Euler - Implicit Scheme
    -------------
    """

    ub = np.zeros((t.size,x.size))


    #Initial condition
    ub[0,:] = g(x)
    ub[0,0] = ub[0,N+1] = 0.0 #Implement boundaries rigidly

    backward_euler(alpha,ub,N,T)



    """
    -------------
    Crank-Nicolson - Implicit Scheme
    -------------
    """

    uc = np.zeros((t.size,x.size))


    #Initial condition
    uc[0,:] = g(x)
    uc[0,0] = uc[0,N+1] = 0.0 #Implement boundaries rigidly

    crank_nicolson(alpha,uc,N,T)


    """
    -------------
    Neural Network PDE-solver
    -------------
    """


    # Defining variables
    num_iter = 100000 # Number of iterations
    num_hidden_neurons = [90] # Number of hidden neurons in each layer
    string = "Adam" # Choosing which Gradient Descent to use, eihter Adam or GD
    learning_rate = 0.0  # When using Adam as GD learning_rate is not needed, but when GD is used a learning_rate needs to be specified


    print("\n The PDE solver you are executing is:")
    print("Deep Neural Network with %d layer" %len(num_hidden_neurons))

    T1, x_mesh, t_mesh, u_dnn = NN_PDE(dx, num_iter, num_hidden_neurons, string, learning_rate)



    """
    -------------
    Analytic solution to PDE
    -------------
    """
    x_analytic, t_analytic = np.meshgrid(x,t)
    x_analytic = x_analytic.ravel()
    t_analytic = t_analytic.ravel()

    analytic = np.exp(-np.pi**2*t_analytic)*np.sin(np.pi*x_analytic)
    Analytic = analytic.reshape((int(1.0/dt),int(1.0/dx)+2))

    """
    -------------
    Plotting
    -------------
    """

    x,t = np.meshgrid(x,t)


    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, t, Analytic, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel('Position $x$')
    plt.ylabel('Time $t$')
    plt.title("Analytic")
    plt.show()


    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, t, uf, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel('Position $x$')
    plt.ylabel('Time $t$')
    plt.title("Forward Euler")
    plt.show()

    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, t, ub, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel('Position $x$')
    plt.ylabel('Time $t$')
    plt.title("Backward Euler")
    plt.show()

    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, t, uc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel('Position $x$')
    plt.ylabel('Time $t$')
    plt.title("Crank-Nicolson")
    plt.show()


    fig = plt.figure();
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x_mesh, t_mesh, u_dnn, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel('Position $x$')
    plt.ylabel('Time $t$')
    ax.set_title('Deep Neural Network with %d layer'%len(num_hidden_neurons))
    plt.show()


    x_analytic = np.linspace(0,L, N+2)
    analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[0][0])
    fig = plt.figure();
    plt.title("Numerical vs analytical solution for t = 0")
    plt.plot(x[0], uf[0], ".")
    plt.plot(x[0], ub[0], ".")
    plt.plot(x[0], uc[0], ".")
    plt.plot(x_mesh[0], u_dnn[0], ".")
    plt.plot(x_analytic, analytic)
    plt.legend(["FE", "BE", "CN", "DNN", "Analytic"])

    plt.show()

    print("\n Time = 0.0")
    print("\n Max absolute difference between Forward Euler & Analytic = ")
    print(np.max(abs(uf[0]-analytic)))

    print("\n Max absolute difference between Backward Euler & Analytic = ")
    print(np.max(abs(ub[0]-analytic)))

    print("\n Max absolute difference between Crank-Nicolson & Analytic = ")
    print(np.max(abs(uc[0]-analytic)))

    print("\n Max absolute difference between Deep Neural Network & Analytic = ")
    print(np.max(abs(u_dnn[0]-analytic)))

    analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[int(T/2)][0])
    fig = plt.figure();
    plt.title("Numerical vs analytical solution for t = 0.5")
    plt.plot(x[0], uf[int(T/2)], ".")
    plt.plot(x[0], ub[int(T/2)], ".")
    plt.plot(x[0], uc[int(T/2)], ".")
    plt.plot(x_mesh[0], u_dnn[int(T1/2)], ".")
    plt.plot(x_analytic, analytic)
    plt.legend(["FE", "BE", "CN", "NN", "Analytic"])
    plt.show()

    print("\n Time = 0.5")
    print("\n Max absolute difference between Forward Euler & Analytic = ")
    print(np.max(abs(uf[int(T/2)]-analytic)))

    print("\n Max absolute difference between Backward Euler & Analytic = ")
    print(np.max(abs(ub[int(T/2)]-analytic)))

    print("\n Max absolute difference between Crank-Nicolson & Analytic = ")
    print(np.max(abs(uc[int(T/2)]-analytic)))

    print("\n Max absolute difference between Deep Neural Network & Analytic = ")
    print(np.max(abs(u_dnn[int(T1/2)]-analytic)))

    analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[T-1][0])
    fig = plt.figure();
    plt.title("Numerical vs analytical solution for t = 1")
    plt.plot(x[0], uf[T-1], ".")
    plt.plot(x[0], ub[T-1], ".")
    plt.plot(x[0], uc[T-1], ".")
    plt.plot(x_mesh[0], u_dnn[T1-1], ".")
    plt.plot(x_analytic, analytic)
    plt.legend(["FE", "BE", "CN", "NN", "Analytic"])
    plt.show()

    print("\n Time = 1.0")
    print("\n Max absolute difference between Forward Euler & Analytic = ")
    print(np.max(abs(uf[T-1]-analytic)))

    print("\n Max absolute difference between Backward Euler & Analytic = ")
    print(np.max(abs(ub[T-1]-analytic)))

    print("\n Max absolute difference between Crank-Nicolson & Analytic = ")
    print(np.max(abs(uc[T-1]-analytic)))

    print("\n Max absolute difference between Deep Neural Network & Analytic = ")
    print(np.max(abs(u_dnn[T1-1]-analytic)))


elif Task == "2":
    """
    -------------
    Solving Eigenvalue Problems of Symmetric Matrices
    -------------
    """

    # Defining variables
    matrix_size = 6 # Size of the matrix
    string1 = "random" # To use a chosen matrix or a random matrix
    run_iter = 1 # Number of times the program will run
    num_iter = 50000 # Number of iterations
    num_hidden_neurons = [100] # Number of hidden neurons in each layer
    string2 = "Adam" # Choosing which Gradient Descent to use, eihter Adam or GD
    learning_rate = 0.0  # When using Adam as GD learning_rate is not needed, but when GD is used a learning_rate needs to be specified

    NN_Eigenvalue(matrix_size, string1, run_iter, num_iter, num_hidden_neurons, string2, learning_rate)



else:
    print("Please write either 1 or 2")
