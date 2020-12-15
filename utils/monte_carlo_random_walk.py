import numpy as np
# import numba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity


""" Monte Carlo Simulation Experminets"""
# inspired by:
# (1) http://justinbois.github.io/bootcamp/2015/lessons/l24_monte_carlo.html
# (2) https://seaborn.pydata.org/generated/seaborn.distplot.html
# (3)
# seaborn settings
sns.set_theme(style="darkgrid")


# ToDo: rewrite Skeleton for our needs
# ToDo: Time Tracking
# ToDo: write experiment results to csv

def monte_carlo_run(n_samples,measure,A,B)
    """
    Do a Monte Carlo simulation of n_samples, plot a histogram, mean and standard deviation
    and write results to csv
    :param n_samples: number of runs
    :param np.array of first parameters used in method, 
    :param np.array of second parameters used in method
    :param measure() The evalutation measure has to be method
    :return: 
    """
    assert callable(measure), \
        " The given measure is not callable"
  
    assert (isinstance(A, np.array) & isinstance(B, np.array)),\
        "A is of type (%s) and B is of type (%s), should be (%s)" % (type(A),type(B), "np.array")
    
    assert (A.ndim == 1 & B.ndim == 1),\
         "A has %d dimensions and B has %d dimensions should be 1" % (A.ndim, B.ndim)

    assert len(A) == len(B),\
         "A has not the same length as B,  %d !=  %d " % (len(A), len(B))

    data_storage = np.empty((len(A),n_samples))

    for i in range(len(A)):
        for j in range(n_samples):
            data_storage[i,j] = measure(A,B)

    mu = np.mean(data_storage, axis= 1)
    sigma = np.std(data_storage, axis= 1)

    plt.figure()
    for i in range(len(A)):
        plt.subplot(int(len(A)/2), 2)   
        sns.distplot(data_storage[i,:])
        print('Mean displacement:', mu[i])
        print('Standard deviation:', sigma[i])
    



def random_walk_1d(n_steps):
    """
    take a random walk of n_steps and return excursion from the origin
    :param n_steps: number of stebern
    :return: sum of steps
    """

    # generate an array of our steps (+1 for right, -1 for left)
    steps = 2 * np.random.randint(0, 2, n_steps) - 1
    # sum of steps
    return steps.sum()

# #steps to take for computing distribution
n_steps = 10000

# number of random walks to take
n_samples = 10000

# initial samples of displacements
x = np.empty(n_samples)

# take all of the random walks
for i in range(n_samples):
    x[i] = random_walk_1d(n_steps)

# make histogram and boxplot visualization for result
ax = sns.distplot(x)


# ax2.set(ylim=(-.5, 10))

x = pd.Series(x, name='$x$ (a.u.)')
sns.displot(x, kde=True)
plt.show()


# trial prints
print(' Mean displacement:', x.mean())
print('Standard deviation:', x.std())






