import matplotlib.pyplot as plot
import numpy as np
import timeit
import math

def rosen(x):
    return 10*(x[1]-x[0]**2)**2 + (1-x[0])**2

def rosenGrad(x):
    return np.array([-2 * (1-x[0]) - 40*x[0]*(x[1]-x[0]**2), 20 * (x[1]-x[0]**2)])
    
def rosenHess(x):
    return np.matrix([[2+80*x[0]**2-40*(-x[0]**2+x[1]), -40*x[0]], [-40*x[0], 20]])

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
            gamma = 9.9 + eig(hessian)[0].min()
            #making the hessian non singular by adding the multiple of the
            #absolute value of the smalled eigen value
            nonSingularHess = hessian + gamma*np.identity(len(hessian))
            p = np.linalg.solve(nonSingularHess, -gradX)

        t = lineSearch(f, gradX, p, x)
        x = x + t * p

        gradX = grad(x)
        norm_gradX = np.linalg.norm(gradX)
        hist.append(x)
    return (x, hist)


def bfgs(f, grad, x0):
    x = x0
    hist = [x0]

    N = len(x)
    B = np.identity(N)

    tol = 1e-6
    eps = 1e-6
    
    norm_gradX = np.linalg.norm(grad(x))
    gradX = grad(x)
    s = np.array([100, 100])
    
    while (norm_gradX > eps) and (np.linalg.norm(s) > tol):
        p = -B.dot(gradX)
        
        t = backtrackLineSearch(rosen, gradX, p, x)
        s = t * p
        x = x + s
        y = grad(x) - gradX

        sdoty = np.dot(s, y)
        s_outer_s = np.outer(s, s)
        By = np.dot(B, y)
        yB = np.dot(y, B)
                
        B1 = (sdoty + np.dot(y, By)) * s_outer_s / sdoty**2
        B2 = (np.outer(By, s) + np.outer(s, yB)) / sdoty
        
        B = B + B1 - B2

        gradX = grad(x)
        norm_gradX = np.linalg.norm(gradX)
        hist.append(x)
        
    return (x, hist)

def plotError(list_f,star_f):
    err = [np.linalg.norm(v) for v in (list_f - star_f)]
    cp = plot.semilogy(err)
    plot.show(cp)
    
    err = err[:-1]
    print(err)
    ratios = [math.log(err[i+1])/math.log(err[i]) for i in range(1, len(err)-1)]
    print(ratios)

(x, hist) = bfgs(rosen, rosenGrad, np.array([0, 0]))
plotError(hist, hist[-1])

"""
BFGS INSIGHT


COMPARISON
BFGS converges for this problem in 14 steps.
Newton converges in 8. This is expected since Newton's method has the exact Hessian.

Timing results:
- BFGS: 0.165772914886 for 100 runs
- Newton: 0.165772914886 for 100 runs
Which accounts for the overhead of calculating the approximation to the hessian in each iteration while the hessian in the Newton method is trivial to compute.

CONVERGENCE RATE
norm(f_k+1 - f_star) < C norm(f_k - f_star)^2
log(err_i+1)/log(err_i) =
[-4.189135971856988, 1.6455537390529578, 1.005088341053911, 1.2167915027203504, 1.794561471919084, 1.1591701412443165, 1.6422573123544812, 1.5145304408087397, 1.491418595177334, 1.223418459584189]
1.5145304408087397, 1.491418595177334, 1.223418459584189
The result makes sense because this method is better than gradient descent (linear) but less performant per iteration from the newton method (quadratic).
"""

def newton():
    (x, hist) = newtonMethod(rosen, rosenGrad, rosenHess, np.array([0, 0]))

print timeit.timeit(newton, number=100)

# print len(hist)
# print x

def bfgs_method():
    (x, hist) = bfgs(rosen, rosenGrad, np.array([0, 0]))

print timeit.timeit(bfgs_method, number=100)

