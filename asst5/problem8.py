from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
# Load the images.
orig_img = Image.open("bwcart.png")

# Convert to arrays.
Uorig = np.array(orig_img)
rows, cols = Uorig.shape

# Known is 1 if the pixel is known,
# 0 if the pixel was corrupted.
# The Known matrix is initialized randomly.
Known = np.zeros((rows, cols))
for i in xrange(rows):
    for j in xrange(cols):
        if np.random.random() > 0.5:
            Known[i, j] = 1

Ucorr = Known*Uorig
corr_img = Image.fromarray(Ucorr)
orig_img = Image.fromarray(Uorig)

# Display the images.
fig, ax = plt.subplots(1, 2,figsize=(10, 5))
ax[1].imshow(corr_img);
ax[1].set_title("Corrupted Image")
ax[1].axis('off');

from cvxpy import *
variables = []
constraints = []
U = Variable(rows, cols)
variables.append(U)
constraints.append(mul_elemwise(Known[:, :], U) == mul_elemwise(Known[:, :], Ucorr[:, :]))
#tv is a built in function l2 total variation defined exacly as the one in the asst
objectiveFunction = tv(*variables)

prob = Problem(Minimize(objectiveFunction), constraints)
prob.solve(verbose=True, solver=SCS)

# Load variable values into a single array.
rec_arr = np.zeros((rows, cols))
rec_arr = variables[0].value

corr_img = Image.fromarray(rec_arr)

ax[0].imshow(corr_img);
ax[0].set_title("Corrected Image")
ax[0].axis('off')

plt.show()
