from cvxpy import *
import numpy as np

n = 5
p = np.array([2, 5, 3, 7, 1])
c = np.array([5, 2, 1, 2, 4])

A = np.matrix([
    [1, 1, 0, 0, 0],
    [-1, 0, 1, 1, 0],
    [0, -1, -1, 0, 1],
    [0, 0, 0, -1, -1]])

q = 4
b = [q, 0, 0, -q]

x = Variable(n)
obj = Minimize(p*x)
constraints = [x >= 0, A*x == b, x <= c]
prob = Problem(obj, constraints)

print "cost", prob.solve()
print x.value
print "node constraints", constraints[1].dual_value
print "capacity constraints", constraints[2].dual_value
