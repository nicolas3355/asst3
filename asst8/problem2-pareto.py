import numpy as np
from cvxpy import *
import matplotlib.pyplot as pyplot

avg = np.array([.12, .10, .07, .03])
dev = np.array([.2, .1, .05, 0])
covariance = np.array([
        [1, .3/2, .4/2, 0],
        [.3/2, 1, 0, 0],
        [.4/2, 0, 1, 0],
        [0, 0, 0, 1]
    ])

X = Variable(4)

constraints = []
constraints.append(sum(X) == 1)
constraints.append(X >= 0)
constraints.append(avg * X >= 0.1)



risk = []
expected = []
for l_value in np.linspace(0, 1, 10):
    objective = Minimize(l_value * quad_form(mul_elemwise(dev, X), covariance)  - avg*X)
    problem = Problem(objective, constraints)
    problem.solve()
    
    risk.append((X.value.T.dot(covariance).dot(X.value))[0, 0])
    expected.append((avg*X.value)[0, 0])

print(risk)
print(expected)

pyplot.plot(expected, risk)
pyplot.show()
