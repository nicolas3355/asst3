import numpy as np
import lib

def f(x):
    (x1, x2, x3, v1, v2) = x[0], x[1], x[2], x[3], x[4]
    return np.array([
        x2 + 2*v1*x1 + 2*v2*x1,
        x1 + x3 + 2*x2*v1,
        x2 + 2*v2*x3,
        x1**2 + x2**2 - 2,
        x1**2 + x3**2 - 2])

def j(x):
    (x1, x2, x3, v1, v2) = tuple(list(  x))
    
    r1 = np.array([2*v1 + 2*v2, 1, 0, 2*x1, 2*x1])
    r2 = np.array([1, 2*v1, 1, 2*x2, 0])
    r3 = np.array([0, 1, 2*v2, 0, 2])
    r4 = np.array([2*x1, 2*x2, 0, 0, 0])
    r5 = np.array([2*x1, 0, 2*x3, 0, 0])
    
    return np.row_stack((r1, r2, r3, r4, r5))

x = lib.newton(f, j, f(np.array([1.0]*5)))[0]
print f(x)
