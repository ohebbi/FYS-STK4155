import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import cm

dxlist = np.array([0.1, 0.01])
dtlist = 0.5*dxlist**2 #Stability criterion
L = 1
T = 1

for i in range(dxlist.size):
    dx = dxlist[i]
    dt = dtlist[i]
    C = dt/(dx**2)

    x = np.linspace(0,L,int(L/dx))
    t = np.linspace(0,T,int(T/dt))
    u_array = np.zeros((t.size,x.size))
    u = np.zeros(len(x))

    u1 = np.sin(np.pi*x) #Initial condition
    u1[0] = u1[-1] = 0.0 #Boundary conditions

    u_array[0] = u1

    for j in range(1, len(t)):
        u[1:-1] = C*(u1[2:] - 2*u1[1:-1] + u1[:-2]) + u1[1:-1]
        u[0] = u[-1] = 0.0
        u_array[j] = u
        u1 = u          #Set previous u for next time step to be current u

    fig = plt.figure();
    x,t = np.meshgrid(x,t)
    ax = fig.gca(projection='3d');
    # Plot the surface.
    surf = ax.plot_surface(x, t, u_array, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False);
                       # Customize the z axis.
    ax.set_zlim(-0.10, 1.40);
    for angle in range(0,150):
        ax.view_init(40,angle)
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Numerical")
    plt.show()

    x_analytic = np.linspace(0,L,10000)
    analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[1][0])
    fig = plt.figure();
    plt.title("Numerical vs analytical solution for t = 0")
    plt.plot(x[0],u_array[1])
    plt.plot(x_analytic,analytic)
    plt.legend(["Numeric", "Analytic"])
    if i == 1:
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d');
        # Plot the surface.
        analytic = np.sin(np.pi*x)*np.exp(-np.pi**2*t)
        surf = ax2.plot_surface(x, t, abs(analytic-u_array), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False);
                           # Customize the z axis.
        ax2.set_zlim(-0.10, 1.40);
        for angle in range(0,150):
            ax2.view_init(40,angle)

        ax2.zaxis.set_major_locator(LinearLocator(10));
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Absolute difference between numerical and analytical solution")
        plt.colorbar(surf)

        plt.show()
