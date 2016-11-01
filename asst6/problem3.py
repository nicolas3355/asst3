from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np

np.random.seed(1)
# Load the images.
orig_img = Image.open("bwcart.png")

#matrix to array
def indexOf(i,j):
    return j + (cols * i)

# Convert to arrays.
Uorig = np.array(orig_img)
rows, cols = Uorig.shape

ejre = np.zeros((rows * cols))
for i in range (0,rows):
    for j in range(0,cols):
        ejre[indexOf(i,j)] = Uorig[i][j]


# Known is 1 if the pixel is known,
# 0 if the pixel was corrupted.
# The Known matrix is initialized randomly.
knownInTwoDimenstion = np.zeros((rows,cols))
Known = np.zeros((rows * cols))
for i in xrange(rows):
    for j in xrange(cols):
        if np.random.random() > 0.5:
            Known[indexOf(i, j)] = 1
            knownInTwoDimenstion[i,j] =1

Ucorr = Known*ejre
UcorruptInTwoDimensions = knownInTwoDimenstion * Uorig
corr_img = Image.fromarray(UcorruptInTwoDimensions)
# orig_img = Image.fromarray(Uorig)

# Display the images.
fig, ax = plt.subplots(1, 2,figsize=(10, 5))
ax[1].imshow(corr_img);
ax[1].set_title("Corrupted Image")
ax[1].axis('off');



# g = np.array([[float(number) for number in rowOfNumbers.split()] \
# for rowOfNumbers in [line.rstrip('\r\n') for line in open('G.dat')]])

imageVector = []
print rows
print cols
for i in xrange(rows):
    for j in xrange(cols):
        imageVector.append(ejre[indexOf(i,j)])

DIFF1 = np.zeros((len(imageVector),len(imageVector)))
DIFF2 = np.zeros((len(imageVector),len(imageVector)))

for i in range (1,rows):
    for j in range(1,cols):
        DIFF1[indexOf(i,j)][indexOf(i,j)] = 1
        DIFF1[indexOf(i,j)][indexOf(i-1,j)] = -1

for i in range (1,rows):
    for j in range(1,cols):
        DIFF2[indexOf(i,j)][indexOf(i,j)] = 1
        DIFF2[indexOf(i,j)][indexOf(i,j-1)] = -1

print "imageVector lenght",len(imageVector)

from cvxpy import *
variables = []
constraints = []
U = Variable(rows * cols)
variables.append(U)
constraints.append(mul_elemwise(Known, U) == mul_elemwise(Known, Ucorr))
#tv is a built in function l2 total variation defined exacly as the one in the asst
objectiveFunction = norm(DIFF1*U,1) + norm(DIFF2*U,1)

prob = Problem(Minimize(objectiveFunction), constraints)
prob.solve(verbose=True, solver=SCS)

# Load variable values into a single array.
rec_arr = np.zeros((rows, cols))
rec_arr = variables[0].value

Corrected_img = np.zeros((rows,cols))
for i in xrange(rows):
    for j in xrange(cols):
        Corrected_img[i][j] = rec_arr[indexOf(i,j)]
corr_img = Image.fromarray(Corrected_img)
#
ax[0].imshow(corr_img);
ax[0].set_title("Corrected Image")
ax[0].axis('off')
plt.show()
