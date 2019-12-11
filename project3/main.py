import numpy, sys, math
from  matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import cm
import numpy as np
import os
from PDE_solvers import *

print("Which step size of Delta x do you want to run")
print("dx = 0.1: Write: '0.1' ")
print("dx = 0.01: Write: '0.01' ")

# Step length in x
dx      =   float(input())

# Number of integration points along x-axis
N       =   int(1.0/dx)

L = 1
#x = numpy.linspace (0,1,N+2)
#t = np.linspace(0,1,T)
#alpha = dt/(dx**2)

#u = np.zeros((t.size,x.size))
#u1 = np.zeros((t.size,x.size))
#u2 = np.zeros((t.size,x.size))
#u3 = np.zeros((t.size,x.size))


#Initial codition
#u[0,:] = g(x)
#u[0,0] = u[0,N+1] = 0.0 #Implement boundaries rigidly




#print "Please select method 1,2, or 3!"


# Define method to use 1 = explicit scheme, 2= implicit scheme, 3 = Crank-Nicolson
print("Which PDE solver do you want to run")
print("Forward Euler - Explicit Scheme: Write 'FE'")
print("Backward Euler - Implicitt Scheme: Write 'BE'")
print("Crank-Nicolson - Implicitt Scheme: Write 'CN'")


#Task = input("Write here: ")

"""
-------------
Forward Euler - Explicit Scheme:
-------------
"""
#if Task == "FE":


# Step length in time
dt      =   0.5*dx*dx
# Number of time steps till final time
T       =   int(1.0/dt)

x = numpy.linspace (0,1,N+2)
t = np.linspace(0,1,T)
alpha = dt/(dx**2)

u1 = np.zeros((t.size,x.size))
u1[0,:] = g(x)
u1[0,0] = u1[0,N+1] = 0.0 #Implement boundaries rigidly


forward_euler(alpha,u1,N,T)

"""
-------------
Backward Euler - Implicitt Scheme
-------------
"""
#if Task == "BE":

# Step length in time
dt1       =   dx
# Number of time steps till final time
T1       =   int(1.0/dt1)

t1 = np.linspace(0,1,T1)
alpha1 = dt1/(dx**2)

u2 = np.zeros((t1.size,x.size))
u2[0,:] = g(x)
u2[0,0] = u2[0,N+1] = 0.0 #Implement boundaries rigidly


backward_euler(alpha1,u2,N,T1)

"""
-------------
Crank-Nicolson - Implicitt Scheme
-------------
"""
#if Task == "CN":


u3 = np.zeros((t1.size,x.size))
u3[0,:] = g(x)
u3[0,0] = u3[0,N+1] = 0.0 #Implement boundaries rigidly

crank_nicolson(alpha1,u3,N,T1)

"""
-------------
Plotting
-------------
"""

x1,t1 = np.meshgrid(x,t1)
x,t = np.meshgrid(x,t)




fig = plt.figure();
ax = fig.gca(projection='3d');
# Plot the surface.
surf = ax.plot_surface(x, t, u1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False);
                   # Customize the z axis.
ax.set_zlim(-0.10, 1.40);
for angle in range(0,150):
    ax.view_init(40,angle)
ax.zaxis.set_major_locator(LinearLocator(10));
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
plt.xlabel('Position $x$')
plt.ylabel('Time $t$')
plt.title("FE")
plt.show()

fig = plt.figure();
ax = fig.gca(projection='3d');
# Plot the surface.
surf = ax.plot_surface(x1, t1, u2, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False);
                   # Customize the z axis.
ax.set_zlim(-0.10, 1.40);
for angle in range(0,150):
    ax.view_init(40,angle)
ax.zaxis.set_major_locator(LinearLocator(10));
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
plt.xlabel('Position $x$')
plt.ylabel('Time $t$')
plt.title("BE")
plt.show()

fig = plt.figure();
ax = fig.gca(projection='3d');
# Plot the surface.
surf = ax.plot_surface(x1, t1, u3, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False);
                   # Customize the z axis.
ax.set_zlim(-0.10, 1.40);
for angle in range(0,150):
    ax.view_init(40,angle)
ax.zaxis.set_major_locator(LinearLocator(10));
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
plt.xlabel('Position $x$')
plt.ylabel('Time $t$')
plt.title("CN")
plt.show()

x_analytic = np.linspace(0,L, N+2)
analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[0][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 0")
plt.plot(x[0],u1[0], ".")
plt.plot(x1[0],u2[0], ".")
plt.plot(x1[0],u3[0], ".")
plt.plot(x_analytic,analytic)
#plt.legend(["Numeric", "Analytic"])
plt.legend(["FE", "BE", "CN", "Analytic"])

plt.show()


analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[int(T/2)][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 0.5")
plt.plot(x[0],u1[int(T/2)], ".")
plt.plot(x1[0],u2[int(T1/2)], ".")
plt.plot(x1[0],u3[int(T1/2)], ".")
plt.plot(x_analytic,analytic)
plt.legend(["FE", "BE", "CN", "Analytic"])
#plt.legend(["Numeric", "Analytic"])
plt.show()


analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[T-1][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 1")
plt.plot(x[0],u1[T-1], ".")
plt.plot(x1[0],u2[T1-1], ".")
plt.plot(x1[0],u3[T1-1], ".")
plt.plot(x_analytic,analytic)
plt.legend(["FE", "BE", "CN", "Analytic"])
#plt.legend(["Numeric", "Analytic"])
plt.show()




"""
X,T = np.meshgrid(x, t)

#x = X.ravel()
#t = T.ravel()
# Analytical solution
x_analytic = np.linspace(0,L,10000)
analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*T[1][0])
#analytic = np.exp(-np.pi**2*t)*np.sin(np.pi*x)

# Plot the results
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Numerical solution')
s = ax.plot_surface(X, T, u1,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(X, T, analytic,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_ylabel('Time $t$')
ax.set_xlabel('Position $x$');



## Take some 3D slices
indx1 = 0
indx2 = int((1/dt)/2)
indx3 = int(1/dt)-1

t1 = t[indx1]
t2 = t[indx2]
t3 = t[indx3]

# Slice the results from the DNN
res1 = u[indx1,:]
res2 = u[indx2,:]
res3 = u[indx3,:]

# Slice the analytical results
res_analytical1 = analytic[indx1,:]
res_analytical2 = analytic[indx2,:]
res_analytical3 = analytic[indx3,:]

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t1)
plt.plot(x, res1, "o")
plt.plot(x,res_analytical1)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x, res2, "o")
plt.plot(x,res_analytical2)
plt.legend(['dnn','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x, res3, "o")
plt.plot(x,res_analytical3)
plt.legend(['dnn','analytical'])

plt.show()
"""
