# Code for solving the 1+1 dimensional diffusion equation
# du/dt = ddu/ddx on a rectangular grid of size L x (T*dt),
# with with L = 1, u(x,0) = g(x), u(0,t) = u(L,t) = 0

import numpy, sys, math
from  matplotlib import pyplot as plt
import numpy as np

def forward_step(alpha, u, uPrev, N):
    """
    Steps forward-euler algo one step ahead.
    Implemented in a separate function for code-reuse from crank_nicolson()
    """

    for x in range(1,N+1): #loop from i=1 to i=N
        u[x] = alpha*uPrev[x-1] + (1-2*alpha)*uPrev[x] + alpha*uPrev[x+1]

def forward_euler(alpha,u,N,T):
    """
    Implements the forward Euler sheme, results saved to
    array u
    """
    #Skip boundary elements
    for t in range(1,T):
        forward_step(alpha, u[t], u[t-1], N)

def tridiag(u, u_prev, alpha, N):
    """
    Tridiagonal gaus-eliminator, specialized to diagonal = 1+2*alpha,
    super- and sub- diagonal = - alpha
    """
    diag = 1+2*alpha
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
    u[0] = 0
    u[N+1] = 0

 #Backward substitute
    for i in range(N, 1, -1):
       u[i] = (u_old[i] - offdiag*u[i+1])/d[i]


def backward_euler(alpha,u,N,T):
    """
    Implements backward euler scheme by gaus-elimination of tridiagonal matrix.
    Results are saved to u.
    """
    #diag = 1 + 2*alpha
    #offdiag = -alpha
    for t in range(1,T):
        tridiag(u[t], u[t-1], alpha, N) #Note: Passing a pointer to row t, which is modified in-place

def crank_nicolson(alpha,u,N,T):
    """
    Implents crank-nicolson scheme, reusing code from forward- and backward euler
    """

    #diag = 2 + 2*alpha/2
    #offdiag = -alpha
    u_temp = u.copy()
    for t in range(1,T):
        #for x in range(1,N+1): #loop from i=1 to i=N
        #    u_temp[t, x] = alpha*u_temp[t-1, x-1] + beta*u_temp[t-1, x] + alpha*u_temp[t-1, x+1]
        forward_step(alpha, u_temp[t], u_temp[t-1], N)
        tridiag(u[t], u_temp[t], alpha, N)


def g(x):
    """Initial condition u(x,0) = g(x), x \in [0,1]"""
    return numpy.sin(math.pi*x)
