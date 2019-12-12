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

# Step length in time
dt      =   0.5*dx*dx
# Number of time steps till final time
T       =   int(1.0/dt)

L = 1
x = numpy.linspace (0,1,N+2)
t = np.linspace(0,1,T)
alpha = dt/(dx**2)








# Define method to use 1 = explicit scheme, 2= implicit scheme, 3 = Crank-Nicolson
print("The PDE solvers you execute are:")
print("Forward Euler - Explicit Scheme")
print("Backward Euler - Implicitt Scheme")
print("Crank-Nicolson - Implicitt Scheme")



"""
-------------
Forward Euler - Explicit Scheme:
-------------
"""
uf = np.zeros((t.size,x.size))


#Initial codition
uf[0,:] = g(x)
uf[0,0] = uf[0,N+1] = 0.0 #Implement boundaries rigidly

forward_euler(alpha,uf,N,T)



"""
-------------
Backward Euler - Implicitt Scheme
-------------
"""

ub = np.zeros((t.size,x.size))


#Initial codition
ub[0,:] = g(x)
ub[0,0] = ub[0,N+1] = 0.0 #Implement boundaries rigidly

backward_euler(alpha,ub,N,T)



"""
-------------
Crank-Nicolson - Implicitt Scheme
-------------
"""

uc = np.zeros((t.size,x.size))


#Initial codition
uc[0,:] = g(x)
uc[0,0] = uc[0,N+1] = 0.0 #Implement boundaries rigidly

crank_nicolson(alpha,uc,N,T)



"""
-------------
Plotting
-------------
"""

x,t = np.meshgrid(x,t)




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
plt.title("FE")
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
plt.title("BE")
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
plt.title("CN")
plt.show()

x_analytic = np.linspace(0,L, N+2)
analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[0][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 0")
plt.plot(x[0], uf[0], ".")
plt.plot(x[0], ub[0], ".")
plt.plot(x[0], uc[0], ".")
plt.plot(x_analytic, analytic)
#plt.legend(["Numeric", "Analytic"])
plt.legend(["FE", "BE", "CN", "Analytic"])

plt.show()

print("\n Time = 0 \n")
print("\n Max difference between Forward Euler & Analytic \n")
print(np.max(abs(uf[0]-analytic)))

print("\n Max difference between Backward Euler & Analytic \n")
print(np.max(abs(ub[0]-analytic)))

print("\n Max difference between Crank-Nicolson & Analytic \n")
print(np.max(abs(uc[0]-analytic)))

analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[int(T/2)][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 0.5")
plt.plot(x[0], uf[int(T/2)], ".")
plt.plot(x[0], ub[int(T/2)], ".")
plt.plot(x[0], uc[int(T/2)], ".")
plt.plot(x_analytic, analytic)
plt.legend(["FE", "BE", "CN", "Analytic"])
#plt.legend(["Numeric", "Analytic"])
plt.show()

print("\n Time = 0.5 \n")
print("\n Max difference between Forward Euler & Analytic \n")
print(np.max(abs(uf[int(T/2)]-analytic)))

print("\n Max difference between Backward Euler & Analytic \n")
print(np.max(abs(ub[int(T/2)]-analytic)))

print("\n Max difference between Crank-Nicolson & Analytic \n")
print(np.max(abs(uc[int(T/2)]-analytic)))

analytic = np.sin(np.pi*x_analytic)*np.exp(-np.pi**2*t[T-1][0])
fig = plt.figure();
plt.title("Numerical vs analytical solution for t = 1")
plt.plot(x[0], uf[T-1], ".")
plt.plot(x[0], ub[T-1], ".")
plt.plot(x[0], uc[T-1], ".")
plt.plot(x_analytic, analytic)
plt.legend(["FE", "BE", "CN", "Analytic"])
#plt.legend(["Numeric", "Analytic"])
plt.show()

print("\n Time = 1 \n")
print("\n Max difference between Forward Euler & Analytic \n")
print(np.max(abs(uf[T-1]-analytic)))

print("\n Max difference between Backward Euler & Analytic \n")
print(np.max(abs(ub[T-1]-analytic)))

print("\n Max difference between Crank-Nicolson & Analytic \n")
print(np.max(abs(uc[T-1]-analytic)))


print("\n Max difference between Forward Euler & Backward Euler \n")
print(np.max(abs(uf-ub)))

print("\n Max difference between Forward Euler & Crank-Nicolson \n")
print(np.max(abs(uf-uc)))

print("\n Max difference between Crank-Nicolson & Backward Euler \n")
print(np.max(abs(uc-ub)))



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
