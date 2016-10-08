import numpy as np
import random
import matplotlib.pyplot as plot
from numpy.linalg import eig

#dimensions
M = 200
N = 4

#starting point
a=[1]*N

#generating sample data
t = np.linspace(0,np.pi/2,M)
p = [0.8*np.sin(2*np.pi*1.15*t[i])+ 1.2 * np.sin(2*np.pi*0.9*t[i]) for i in range(200)]
random.seed(73)
sampleData = [p[i] + random.uniform(-0.15,0.15) for i in range(M)]

def plotNoisyFunction():
    plot.plot(sampleData)
    # plot.show()

def model(a,t):
    return a[0]*np.sin(2*np.pi*a[1]*t) + a[2]*np.sin(2*np.pi*a[3]*t)

def fi(x):
    r = [ model(x,t[i]) - sampleData[i] for i in range(len(sampleData))]
    r = np.array(r)
    return r

def objectiveFunction(x):
    r = fi(x)
    return 0.5 * (r.dot(r))

def jacobian(x):
    jacobian = np.zeros((M,N))
    const = 2*np.pi

    for i in range(M):
        for j in range(N):
            if(j == 0):
                jacobian[i][j] = np.sin(2*np.pi*a[1]*t[i])
            elif(j==1):
                jacobian[i][j] = x[0] * np.cos(const*x[1]*t[i])*const*t[i]
            elif(j==2):
                jacobian[i][j] = np.sin(2*np.pi*a[3]*t[i])
            else:
                jacobian[i][j] = x[2] * np.cos(const*x[3]*t[i])*const*t[i]

    return jacobian

def gradient(x):
    return fi(x).dot(jacobian(x))

def gauss_hessian(x):
    jacobianAtx = jacobian(x)
    return (jacobianAtx.T).dot(jacobianAtx)

def backtrackLineSearch(f, gk, pk, xk):
    a = .1
    b = .8

    t = 1
    # armijo-goldstein condition
    while f(xk+t*pk) > f(xk) + a * t * np.dot(gk, pk):
        t = b * t

    return t

def newtonMethod(f, grad, hess, x0, lineSearch = backtrackLineSearch):
    x = x0
    hist = [x0]
    tol = 1e-5
    norm_gradX = np.linalg.norm(grad(x))
    gradX = grad(x)

    while not (norm_gradX < tol):
        #the system of linear equation that needs to be solved to determine the direction
        hessian = hess(x)
        try:
            p = np.linalg.solve(hessian, -gradX)
        except np.linalg.linalg.LinAlgError:
            #smallest eigenvalue of our hessian
            gamma = eig(hessian)[0].min()
            #making the hessian non singular by adding the multiple of the
            #absolute value of the smalled eigen value
            nonSingularHess = hessian + 1.5*np.abs(gamma)*np.identity(len(hessian))
            p = np.linalg.solve(nonSingularHess, -gradX)

        t = lineSearch(f, gradX, p, x)
        x = x + t * p

        gradX = grad(x)
        norm_gradX = np.linalg.norm(gradX)
        hist.append(x)
    return (x, hist)

def plotfixedFunction():
    (x, hist) = newtonMethod(objectiveFunction,gradient,gauss_hessian,a)
    fixedPoints = [model(x,t[i]) for i in range(len(sampleData))]
    #print fixedPoints
    plot.plot(fixedPoints)
    plot.show()

plotNoisyFunction()
plotfixedFunction()
