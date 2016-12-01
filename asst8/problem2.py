import numpy as np
from cvxpy import *

avg = np.array([.12, .10, .07, .03])
dev = np.array([.2, .1, .05, 0])
covariance = np.array([
        [1, .3/2, .4/2, 0],
        [.3/2, 1, 0, 0],
        [.4/2, 0, 1, 0],
        [0, 0, 0, 1]
    ])

X = Variable(4)
objective = Minimize(quad_form(mul_elemwise(dev, X), covariance))

constraints = []
constraints.append(sum(X) == 1)
constraints.append(X >= 0)
constraints.append(avg * X >= 0.1)


problem = Problem(objective, constraints)
problem.solve(verbose=True)
print("X", X.value)
print("L", [c.dual_value for c in constraints])

######
# ('X', matrix([[  2.22019327e-01],
#         [  6.29967791e-01],
#         [  1.48012881e-01],
#         [  9.65240119e-10]]))
# ('L', [0.027079342405003835, matrix([[  7.67490736e-11],
#         [  2.52054688e-11],
#         [  1.15479074e-10],
#         [  1.47761115e-02]]), 0.41010769915358669])
