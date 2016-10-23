import numpy as np

def backtrackLineSearch(f, gk, pk, xk):
    a = .1
    b = .8

    t = 1
    # armijo-goldstein condition
    while f(xk+t*pk) > f(xk) + a * t * np.dot(gk, pk):
        t = b * t

    return t

def newton(f, j, x0):
    tol = 1e-9
    x = x0
    hist = [x0]
    
    def metric(x):
        fx = f(x)
        return 0.5 * fx.dot(fx)
    
    def metric_grad(x):
        return j(x).dot(f(x))
    
    v = f(x)
    while np.linalg.norm(v) > tol:
        p = np.linalg.solve(j(x), -f(x))
        t = backtrackLineSearch(metric, metric_grad(x), p, x)
        dx = t * p
        x = x + dx
        v = f(x)
    
    return (x, hist)
