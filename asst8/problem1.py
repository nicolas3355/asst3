import numpy as np
from cvxpy import *

A1 = np.array([
   [5.8479532e-01,  -1.9354839e+00],
   [2.3859649e+00,   1.1428571e+00],
   [7.0175439e-01,   1.2350230e+00],
   [-1.0292398e+00,   6.8202765e-01],
   [-1.1695906e+00,   5.5299539e-02],
   [-1.4736842e+00,  -1.1797235e+00]
  ])

b1 = np.array([
   3.3837281e+00,
   9.5981890e-01,
   1.1496483e+00,
   2.4695071e+00,
   2.3474816e+00,
   3.6227127e+00,
])

A2 = np.array([
  [ 7.0175439e-02,  -2.2304147e+00],
  [ 2.4795322e+00,   5.5299539e-02],
  [ 1.1228070e+00,   1.6774194e+00],
  [-1.0994152e+00,   1.1244240e+00],
  [-2.5730994e+00,  -6.2672811e-01]
  ])

b2 = np.array([
  -6.9765812e-01,
   9.0161964e+00,
   8.8853316e+00,
   2.4482712e+00,
  -3.8164228e+00,
  ])

X1 = Variable(2)
X2 = Variable(2)

constraints = []
constraints.append(A1 * X1 <= b1)
constraints.append(A2 * X2 <= b2)

problem = Problem(Minimize(norm(X1-X2)), constraints)
problem.solve(verbose=True)
print("X1", X1.value)
print("X2", X2.value)
print("Dual1", constraints[0].dual_value)
print("Dual2", constraints[1].dual_value)

# RESULTS

# ('X1', matrix([[ 0.44879957],
#         [-0.09712599]]))
# ('X2', matrix([[ 1.39631338],
#         [ 0.35672515]]))
# ('Dual1', matrix([[  1.51246364e-11],
#         [  3.77992800e-01],
#         [  1.22984358e-11],
#         [  4.96226957e-12],
#         [  4.79581079e-12],
#         [  3.65192630e-12]]))
# ('Dual2', matrix([[  9.44700117e-02],
#         [  3.28240659e-12],
#         [  1.58101511e-12],
#         [  4.76459039e-12],
#         [  3.53078870e-01]]))


# X1 is on an edge since only one constraints is active.
# X2 is on a vertex since two constraints are active.