from math import pi
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
# import cvxpy
import random

import matplotlib.pyplot as pyplot

n = 1000
t = np.linspace(0, 0.04, n)
y = np.sin(2*pi*697*t) + np.sin(2*pi*1209*t)/2
D = dct(np.eye(n), type=1, axis=0)
Dy = D.dot(y)
dctY = dct(y)

known = [0]*n
for i in range(n):
    if random.random() < 0.5:
        known[i] = 1.0
knownMat = np.diag(np.array(known))

from cvxpy import *
variables = []
constraints = []
X = Variable(n)
variables.append(X)
constraints.append(knownMat.dot(D) * X == known * y)
objective1 = sum_squares(X)
objective2 = norm(X, 1)
problem = Problem(Minimize(objective2), constraints)
problem.solve(verbose=True, solver=SCS)

output = np.squeeze(np.asarray(X.value))
pyplot.plot(y)
pyplot.plot(output)
# pyplot.plot(np.multiply(known, y))
pyplot.show()
