import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import norm



#number of points
segments = 100
n = segments - 1
u = [0] * n
uStart = 2.0
uEnd = 3.0
l = 5.0
c = 0.5
h = l/segments

def d2u(u,i):
    if (i == 0):
        u0 = uStart
        u1 = u[i+1]
    elif i == n-1:
        u0 = u[i-1]
        u1 = uEnd
    else:
        u0 = u[i-1]
        u1 = u[i+1]
    return (-1/h**2)*(-u0 + 2*u[i] - u1)

def du(u,i):
    if (i == 0):
        u0 = uStart
        u1 = u[i+1]
    elif i == n-1:
        u0 = u[i-1]
        u1 = uEnd
    else:
        u0 = u[i-1]
        u1 = u[i+1]
    return (1.0/(2*h)) * (u1 - u0)

def rightHand(u,i):
    return c*((1+du(u,i)**2)**0.5)

def f(u):
    b = u[:]
    b = [ d2u(u,i)-rightHand(u,i) for i in range(0,len(u))]
    return np.array(b)

def jacobian(u):
    jacobian = np.zeros([n,n])
    for i in range(0,n):
        if(i == 0):
            jacobian[i][i] = -(2 / h**2)
            jacobian[i][i+1] = (1 / h**2) - (c/4/h**2*(u[i+1]-uStart))/rightHand(u,i)
        elif(i == n - 1):
            jacobian[i][i-1] = (1 / h**2) + (c/4/h**2*(uEnd-u[i-1]))/rightHand(u,i)
            jacobian[i][i] = -(2 / h**2)
        else:
            jacobian[i][i-1] = (1 / h**2) + (c/4/h**2*(u[i+1]-u[i-1]))/rightHand(u,i)
            jacobian[i][i] = -(2 / h**2)
            jacobian[i][i+1] = (1 / h**2) - (c/4/h**2*(u[i+1]-u[i-1]))/rightHand(u,i)
    return jacobian

from problem2 import globalizedNewton as nt

#plotting .......
x = nt(f,jacobian,np.array([0]*n))[0]
xWithBoundaries = [uStart] + list(x) + [uEnd]
import matplotlib.pyplot as plot
t = np.linspace(0,l,n+2)
plot.plot(t,xWithBoundaries)
plot.show()
