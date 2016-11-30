from math import pi
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
from numpy.linalg import norm
import matplotlib.pyplot as pyplot
import random
from cvxpy import *

##################################################
# initializing data
##################################################

n = 2500
t = np.linspace(0, 0.04, n)
y = np.sin(2*pi*697*t) + np.sin(2*pi*1209*t)/2
D = dct(np.eye(n), type=1, axis=0)


################################################################################
# Mark 10% of the original data as known
################################################################################
known = [0]*n
for i in range(n):
    if random.random() <= 0.1:
        known[i] = 1.0
knownMat = np.diag(np.array(known))



################################################################################
# genrate the constraints
################################################################################
def constraints(X):
    constraints = []
    constraints.append(knownMat.dot(D) * X == known * y)
    return constraints

################################################################################
# using L2 norm
################################################################################

def generateL2NormObjective(X):
    return sum_squares(X)

################################################################################
# using L1 norm
################################################################################
def generateL1NormObjective(X):
    return norm(X, 1)

################################################################################
#cvx Solver
################################################################################
def solveAndPlot(constraints,objective):
    variables = []
    X = Variable(n)
    variables.append(X)
    problem = Problem(Minimize(objective(X)), constraints(X))
    problem.solve(verbose=True, solver=SCS)

    output = np.squeeze(np.asarray(X.value))
    #reverse fourier to get the original signal
    output = idct(output)
    pyplot.plot(y)
    pyplot.plot(output)
    pyplot.show()



################################################################################
# uncomment the following line to reconstruct using L2 Norm
################################################################################
#solveAndPlot(constraints,generateL2NormObjective)

################################################################################
# uncomment the following line to reconstruct using L1 Norm
################################################################################
solveAndPlot(constraints,generateL1NormObjective)


################################################################################
#L1 norm is better than L2 infact, L1 norm reproduces the original waveform better
#than L2, you can see that from the plot
#unfortunately i couldn't hear any of the 2 waves as the sample size is less than
#44100 (i can generate a .wave raw)
#increasing n to be more than 44100 will introduce memory error and the code will
#more complexe
################################################################################
