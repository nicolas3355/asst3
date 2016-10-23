import numpy as np
'''
    minimize x1-x2
    subject to x1x2 + 4 = 0
'''

#lagrange function of
def f(x):
    x1 , x2 , lamda = list(x)
    return x1 - x2 +  lamda * (x1*x2 + 4)


#KKT first order optimality condition
def gradf(x):
    x1 , x2 , lamda = list(x)
    return np.array([1+lamda*x2,\
                    -1 + lamda *x1,\
                     x1*x2 + 4])
def getLamda(x1,x2):
    return -1.0/x2

def isFirstOrderOptimalityConditionSatisfied(x1,x2):
    lamda = getLamda(x1,x2)
    #result contains an array of boolean comparing each number
    #in the array at a time
    result = ( gradf([x1,x2,lamda]) == np.array([0,0,0]) )
    for result in result:
        if( not result):
            return False
    return True

def jacobian(x):
    x1 , x2 , lamda = list(x)
    return np.array([\
                    [0,lamda,x2],\
                    [lamda,0,x1],\
                    [x2,x1,0]\
                    ])

# z is the null space of the constraints
Z = np.array([1,1])

def hessian(x):
    lamda = x[2]
    return np.array([\
                    [0,lamda],\
                    [lamda,0]
                    ])

# ZT * A * Z need to be postive definite
def isValidSolution(x1,x2):
    lamda = getLamda(x1,x2)
    return (Z.dot(hessian([x1,x2,lamda]))).dot(Z) > 0


def printResults(x1,x2):
    print "is x1 = " ,x1, "and x2 =", x2," satisify the first order optimality condition?:"
    print isFirstOrderOptimalityConditionSatisfied(x1,x2)
    print "is this a valid solution?"
    print isValidSolution(x1,x2)
    print "#####################################"

printResults(2,-2)
printResults(-2,2)
