import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plot

x = cvx.Variable(2)
R = cvx.Variable(1)
a = np.array([[0,-1],[2,-1],[1,1],[-1.0/3,1],[-1,0],[-1,-1]])
b = np.array([0,9,7,3,0,-1])

for i in range(len(b)):
    dir = a[i] / np.linalg.norm(a[i])
    centralPoint = a[i]/np.linalg.norm(a[i])*b[i]
    rightP = centralPoint + np.array([-dir[1], dir[0]])*10
    leftP = centralPoint - np.array([-dir[1], dir[0]])*10
    plot.plot([leftP[0], rightP[0]], [leftP[1], rightP[1]])

objective  = cvx.Maximize(R)

constraints = [(a[i]*x - b[i]) <= -R*np.linalg.norm(a[i]) for i in range(len(b))]

prob = cvx.Problem(objective,constraints)

result = prob.solve()
print x.value
print R.value

plot.plot([x.value[0][0,0] + np.cos(3.141592654*2/39*i)*R.value for i in range(40)], [x.value[1][0,0] + np.sin(3.141592654*2/39*i)*R.value for i in range(40)])
plot.show()
for i in range(0,len(constraints)):
    print constraints[i].dual_value
