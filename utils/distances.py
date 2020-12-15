import numpy as np

############################ v0.01 #######################
"""
util function for different distance measure
"""


def md(A, Vhat):
    """Minimum distance index as defined in
    P. Ilmonen, K. Nordhausen, H. Oja, and E. Ollila.
    A new performance index for ICA: Properties, computation and asymptotic
    analysis.
    In Latent Variable Analysis and Signal Separation, pages 229–236. Springer,
    2010.

    This Code is from the coroICA package which implements the coroICA algorithm presented in 
    "Robustifying Independent Component Analysis by Adjusting for Group-Wise Stationary Noise" 
    by N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf.
    - https://github.com/sweichwald/coroICA-python
    """
    d = np.shape(A)[0]
    G = Vhat.dot(A)
    Gsq = np.abs(G)**2
    Gtilde = Gsq / (Gsq.sum(axis=1)).reshape((d, 1))
    costmat = 1 - 2 * Gtilde + \
        np.tile((Gtilde**2).sum(axis=1), d).reshape((d, d))
    row_ind, col_ind = linear_sum_assignment(costmat)
    md = np.sqrt(d - np.sum(np.diag(Gtilde[row_ind, col_ind]))) / \
        np.sqrt(d - 1)
    return md
