import numpy as np

def f(x):
    y1 = (2*x[0] + x[1]) / (1 + (2*x[0] + x[1])**2)**0.5
    y2 = (2*x[0] - x[1]) / (1 + (2*x[0] - x[1])**2)**0.5
    return np.array([y1, y2])
    
def jacobian(x):
    df1_dx = - 2*(2*x[0] + x[1])**2 / (1 + (2*x[0] + x[1])**2)**1.5 \
        + 2/(1 + (2*x[0] + x[1])**2)**0.5
    df1_dy = df1_dx / 2
    
    df2_dx = 2/(1 + (2*x[0] - x[1])**2)**0.5 \
        - 2*(2*x[0] - x[1])**2 / (1 + (2*x[0] - x[1])**2)**1.5
    
    df2_dy = -1/(1 + (2*x[0] - x[1])**2)**0.5 \
        + (2*x[0] - x[1])**2 / (1 + (2*x[0] - x[1])**2)**1.5
    
    return np.column_stack([
        np.array([df1_dx, df2_dx]),
        np.array([df1_dy, df2_dy]) ])

def naiveNewton(f, j, x0):
    tol = 1e-6
    x = x0
    hist = [x0]
    
    v = f(x)
    while np.linalg.norm(v) > tol:
        print x, f(x)
        dx = np.linalg.solve(j(x), -f(x))
        x = x + dx
        v = f(x)
    
    return (x, hist)

print naiveNewton(f, jacobian, np.array([0.3, 0.3]))
# print naiveNewton(f, jacobian, np.array([0.5, 0.5]))

def backtrackLineSearch(f, gk, pk, xk):
    a = .1
    b = .8

    t = 1
    # armijo-goldstein condition
    while f(xk+t*pk) > f(xk) + a * t * np.dot(gk, pk):
        t = b * t

    return t

def globalizedNewton(f, j, x0):
    tol = 1e-6
    x = x0
    hist = [x0]
    
    def metric(x):
        fx = f(x)
        return 0.5 * fx.dot(fx)
    
    def metric_grad(x):
        return j(x).dot(f(x))
    
    v = f(x)
    while np.linalg.norm(v) > tol:
        print x, f(x)
        p = np.linalg.solve(j(x), -f(x))
        t = backtrackLineSearch(metric, metric_grad(x), p, x)
        dx = t * p
        x = x + dx
        v = f(x)
    
    return (x, hist)

print globalizedNewton(f, jacobian, np.array([0.5, 0.5]))