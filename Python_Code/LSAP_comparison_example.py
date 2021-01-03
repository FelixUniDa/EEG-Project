from itertools import combinations
import numpy as np
from numpy.random import choice, randn, randint
from numpy.linalg import inv
from scipy.optimize import linear_sum_assignment

def LASP_comparison():
    d = choice([5, 25, 50])
    A = randn(d, d) ** randint(1, 3)
    V = inv(A)
    Vhat = V + choice([0, .1, 1]) * randn(d, d) * np.diag(np.arange(1, d + 1))
    G = Vhat.dot(A)
    Gtilde = (G**2) / ((G**2).sum(axis=1)).reshape((d, 1))
    # 4 variants to set up the costmats
    costmats = [
        # for the first 2 variants, the (i,t)th entry..
        # ..captures how well column I[:, i] and G[:, j] align
        1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=0), d).reshape((d, d)),
        # ..captures how well row I[i, :] and G[j, :] align (resembles the paper)
        1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=1), d).reshape((d, d)).T,
        # the latter 2 variants are more indirect and the sum of
        # the d entries selected for a candidate permutation matrix
        # reflects how well the..
        # ..columns..
        1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=0), d).reshape((d, d)).T,
        # ..rows..
        1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=1), d).reshape((d, d)),
        # ..align between the identity and Gtilde
        ]
    inds = []
    for costmat in costmats:
        inds.append(linear_sum_assignment(costmat))
    allindsequal = np.all([
        np.all(np.c_[inds[i]] == np.c_[inds[i + 1]])
        for i in range(3)])
    return allindsequal

same = np.mean([LASP_comparison() for _ in range(5000)]) * 100

print(f'the 4 variants resulted in the same result {same}% of times')
# the 4 variants resulted in the same result 100.0% of times
