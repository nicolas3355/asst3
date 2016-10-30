import numpy as np
from scipy.optimize import linprog

#FORMULATION
c = np.array([0.5, 0.8])
A = np.row_stack([[3, 0], [2, 4], [2, 5]])
b = np.array([6, 10, 8])

"""
Variables: x = [# of brownies, # of cheesecake]
Minimize cost to buy x: f = [cost of brownie, cost of cheesecase] dot x
Constraints: A x > b, where A encodes the constituents in chocolate, sugar, and cream cheese in each column for each of the products.
A x is the 'nutrition' content, b is the minimum nutrition required.

linprog requires the inequality constraints to be provided as upper bounds, which is why in the call below we are providing -A and -b.
"""


# SOLVE
res = linprog(c, A_ub=-A, b_ub=-b)
print res

xstar = res['x']
lstar = res['slack']

print xstar
print lstar

# returns
"""
    fun: 2.2
message: 'Optimization terminated successfully.'
    nit: 3
  slack: array([ 0. ,  0. ,  3.5])
 status: 0
success: True
      x: array([ 2. ,  1.5])
"""

# Lagrange Multipliers
"""
Only the cream cheese constraint is active.
If the negative of the required cream cheese increases by epsilon, as in, the minimum cost will decrease by 3.5 epsilon. 3.5 dollars for the minimum cost per additional ounce of cream cheese.
"""
def lagrange(x, l):
    return c.dot(x) + l.dot(-A.dot(x)+b)

# feasibility:
print "Feasibility", -A.dot(xstar)+b
# prints [ 0.   0.  -3.5], all non-positive

# complementary condition: fi dot lambdas = 0
"""
lambdas = [0.0333, 0.20, 0.0]

so lambda * feasibility_vector = [0, 0, 0]

"""

c = np.array([0.5, -0.5, 0.8, -0.8, 0, 0, 0])
A = np.matrix([
    [-3, 3, 0, 0, 1, 0, 0],
    [-2, 2, -4, 4, 0, 1, 0],
    [-2, 2, -5, 5, 0, 0, 1]])

res = linprog(c, A_eq=A, b_eq=-b)
print "Canon", res
"""
Canon
     fun: 2.2000000000000002
 message: 'Optimization terminated successfully.'
     nit: 3
   slack: array([], dtype=float64)
  status: 0
 success: True
       x: array([ 2. ,  0. ,  1.5,  0. ,  0. ,  0. ,  3.5])
"""

