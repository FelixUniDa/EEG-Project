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






