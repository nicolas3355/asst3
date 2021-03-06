import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import norm
d = np.array([float(line.rstrip('\r\n')) for line in open('d.dat')])
g = np.array([[float(number) for number in rowOfNumbers.split()] \
for rowOfNumbers in [line.rstrip('\r\n') for line in open('G.dat')]])

#computing the eigenValues of g shows that we have eigenValues that are almost 0
smallesEigenValue = eig(g)[0].min()
if(smallesEigenValue < 10e-15):
    print "we have very small eigenValues they are almost 0 for the matrix G"


#moreover computing the inverse of G gives that an execption saying G is singular
try:
    inv(g)
except np.linalg.linalg.LinAlgError as e:
    print e

#g.T*g x = At * b
matrix = ((g.T).dot(g))

#fetching the eien vale of g.gT
eigenValues = eig(matrix)[0]
if(eigenValues.max() / eigenValues.min() < 10e-15):
    print "matrix G transpose times G has very small cappa!!!"


#moreover computing the inverse of g.T*g gives that an execption saying g.T*g is singular
try:
    inv(matrix)
except np.linalg.linalg.LinAlgError as e:
    print e

alpha = 0.001

x = np.linalg.solve(matrix + alpha*np.identity(len(matrix)),(g.T).dot(d))
img = np.zeros((16,16))
for i in range(16):
    for j in range(16):
        img[i,j] = x[i + 16*j]

import matplotlib.pyplot as plot
# import matplotlib.image as mpimg

plot.imshow(img)
plot.show()

print x
