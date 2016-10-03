import numpy as np

#dimensions
M = 200
N = 4

#parameter array
a=[1]*N

#random sample data
sampleData = [0]*M

def model(a,t):
    return a[0]*np.sin(2*np.pi*a[1]*t) + a[2]*np.sin(2*np.pi*a[3]*t)

def fi(x):
    r = [ model(x,i) - sampleData[i] for i in range(len(sampleData))]
    r = np.array(r)
    return r

def objectiveFunction(x):
    r = fi(x)
    return 0.5 * (r.dot(r))

def jacobian(x):
    jacobian = np.zeros((M,N))
    #df/da1 = a1
    #df/da2 = a1 cos(2pia2t) * a2
    #df/da3 = a3
    #df/da4 = a3 cos(2pia4t) * a4
    const = 2*np.pi

    for i in range(M):
        for j in range(N):
            if(j == 0):
                jacobian[i][j] = x[0]
            elif(j==1):
                jacobian[i][j] = x[0] * np.cos(const*i*x[1])*x[1]
            elif(j==2):
                jacobian[i][j] = x[2]
            else:
                jacobian[i][j] = x[2] * np.cos(const*i*x[3])*x[3]

    return jacobian

def gradient(x):
    return fi(x).dot(jacobian(x))

def gauss_hessian(x):
    jacobian = jacobian(x)
    return jacobian.dot(jacobian)

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

(x, hist) = newtonMethod(objectiveFunction,gradient,gauss_hessian,a)
plotError(hist,hist[len(hist)-1])
