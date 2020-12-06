#%%
import numpy as np

# %%
def center(x):
    """Centers input data x by subtracting the mean of each Signal represented by the row vectors of x.

    Args:
        x ([type]): [description]

    Returns:
        centered [array]: Centered data array.
        mean [array]: Vector containing the mean of each signal.
    """
    #mean = np.mean(x, axis=1, keepdims=True)
    mean = np.apply_along_axis(np.mean,axis=1,arr=x)
    centered =  x
    n,m = np.shape(x)
    for i in range(0,n,1):
        centered[i,:] = centered[i,:]-mean[i]
    #print(centered)
    return centered, mean

# %% [markdown]

# %%
def covariance(x):
    """Calculate Covariance matrix for the rowvectors of the input data x.

    Args:
        x (array): Mixed input data.

    Returns:
        cov [array]: Covariance Matrix of the signals represented by the rowvectors of x.
    """
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean
    cov = (m**2)/n
    return cov


def whiten(x):
    """linearly transform the observed signals X in a way that potential 
       correlations between the signals are removed and their variances equal unity. 
       As a result the covariance matrix of the whitened signals will be equal to the identity matrix

    Args:
        x (array): Centered input data.

    Returns:
        x_whitened: Whitened Data with covariance matrix that is equal to identity matrix.
    """
    centered_X, _ = center(x)

    cov = np.cov(centered_X,bias=True) #calculate Covariance matrix between signals

    d, E = np.linalg.eig(cov) #Eigenvalue decomposition (alternatively one can use SVD for whitening)
    idx = d.argsort()[::-1]   
    d = d[idx]                #Sort eigenvalues
    E = E[:,idx]              #Sort eigenvectors

    D_inv = np.diag(1/np.sqrt(d))   #Calculate D^(-0.5)

    W_whiten =  D_inv @ E.T         #Calculate Whitening matrix W_whiten 
    x_whitened = (W_whiten @ centered_X)

    return x_whitened
    