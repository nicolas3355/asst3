import numpy as np
import matplotlib.pyplot as plot

a = [1]*10
h =  0.75/(10+1)
r0 = 1
K = 2*np.pi*h

def objectiveFunction(x):
    return K*x[0]*fi(x[0],r0,h) + K*sumOfFi(x,h) + K*x[-1]*fi(x[-1],r0,h)

# x is a vector
def sumOfFi(x,h):
        return sum (x[i]*fi(x[i],x[i+1],h) for i in range(len(x)-1))


#x1 x2 are values
def fi(x1,x2,h):
    return (1+((x2-x1)/h)**2)**0.5


def grad(x):
    vectorGrad = [0.0]*len(x)
    c0 = (1 + ((r0-x[0])/h)**2)**0.5
    vectorGrad[0] = c0 - ((r0-x[0])*x[0]) / ((h**2)*c0)
    c1 = (1 + ((r0-x[-1])/h)**2)**0.5
    vectorGrad[-1] = c1 - ((r0-x[-1])*x[-1] )/ ((h**2)*c1)

    for i in range (0,len(x)-1):
        tmp_i,tmp_i1 = d_fi(x[i],x[i+1])
        vectorGrad[i] += tmp_i
        vectorGrad[i+1] += tmp_i1

    return np.array(vectorGrad)*K

def hess(x):
    N = len(x)
    result = np.zeros((N,N))

    result[0,0]=\
        -((2*(r0 - x[0]))/(h**2 * (1 + ((r0 - x[0])**2)/h**2)**0.5)) + x[0]/( \
        h**2 *(1 + (r0 - x[0])**2/h**2)**0.5) - ((r0 - x[0])**2 * x[0])/( \
        h**4 *(1 + ((r0 - x[0])**2)/h**2)**(1.5))

    result[-1,-1] = -((2*(r0 - x[-1]))/(h**2 * (1 + ((r0 - x[-1])**2)/h**2)**0.5)) + x[-1]/( \
        h**2 *(1 + (r0 - x[-1])**2/h**2)**0.5) - ((r0 - x[-1])**2 * x[-1])/( \
        h**4 *(1 + ((r0 - x[-1])**2)/h**2)**(1.5))


    # result[0,0] = (-2 * (r0 - x[0])**3 + h**2 * (3*x[0]-2*r0)) / h**4 / ((h**2 + (r0-x[0])**2/h**2))**1.5
    # result[-1,-1] = (-2 * (r0 - x[-1])**3 + h**2 * (3*x[-1]-2*r0)) / h**4 / ((h**2 + (r0-x[-1])**2/h**2))**1.5

    for i in range(0, len(x)-1):
        localHess = d2_fi(x[i], x[i+1])
        result[i,i] += localHess[0][0]
        result[i,i+1] += localHess[0][1]
        result[i+1,i] += localHess[1][0]
        result[i+1,i+1] += localHess[1][1]

    return result*K


def d_fi(x0,x1):
    c = (1 + ((x0-x1)/h)**2)**0.5
    return ( -(x0*(-x0+x1))/((h**2)*c) + c, x0*(-x0+x1)/((h**2)*c) )

def d2_fi(x0,x1):
    result = np.zeros((2,2))
    c = (h**2 + (x0 - x1)**2) / h**2

    d2_x0_x0 = (h**2 * (3*x0 - 2*x1) + 2*(x0 - x1)**3) / h**4 / c**1.5
    d2_x0_x1 = (-(x0-x1)**3 + h**2 * (x1-2*x0)) / h**4 / c**1.5
    d2_x1_x1 = x0 / h**2 / c**1.5

    val = ((d2_x0_x0, d2_x0_x1), (d2_x0_x1, d2_x1_x1))
    # print("local hess", val)
    return val

def verify():
    eps = 0.00001

    b = [1] * len(a)
    b[-1] = a[-1] + eps
    print ("computed grad", (objectiveFunction(b)-objectiveFunction(a))/eps)

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
        p = np.linalg.solve(hess(x), -gradX)
        t = lineSearch(f, gradX, p, x)
        x = x + t * p

        gradX = grad(x)
        norm_gradX = np.linalg.norm(gradX)
        hist.append(x)
    return (x, hist)

def plotError(list_f,star_f):
    err = [np.linalg.norm(v) for v in (list_f - star_f)]
    cp = plot.semilogy(err)
    plot.show(cp)

(x, hist) = newtonMethod(objectiveFunction,grad,hess,a)
plotError(hist,hist[len(hist)-1])
