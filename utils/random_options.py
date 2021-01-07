import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import Python_Code.PowerICA_Python.PowerICA as PICA


# setting random seed
np.random.seed(41)

# Sampling is done with replacement by default
np.random.choice(4, 12)

# Probability weights can be given
np.random.choice(4, 12, p=[.4, .1, .1, .4])
x = np.random.randint(0, 10, (8, 12))

# sampling individual elements
np.random.choice(x.ravel(), 12)

# sampling rows
idx = np.random.choice(x.shape[0], 4)

print("Example Rows: \n" + str(x[idx, :]))

# sampling columns
idx = np.random.choice(x.shape[1], 4)
print("Example Colums: \n" + str(x[:, idx]))

#### Sampling Without replacement
# Give the argument replace=False
try:
    np.random.choice(4, 12, replace=False)
except ValueError as e:
    print(e)

# Shuffling occurs "in place" for efficiency
np.random.shuffle(x)
# To shuffle columns instead, transpose before shuffling
np.random.shuffle(x.T)
# numpy.random.permutation does the same thing but returns a copy
np.random.permutation(x)
# When given an integer n, permutation treats is as the array arange(n)
np.random.permutation(10)
# Use indices if you needed to shuffle collections of arrays in synchrony
x = np.arange(12).reshape(4,3)
y = x + 10
idx = np.random.permutation(x.shape[0])
list(zip(x[idx, :], y[idx, :]))