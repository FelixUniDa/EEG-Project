#%%
import numpy as np

# %%
def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered =  x - mean 
    return centered, mean

# %% [markdown]
# For the second pre-processing technique we need to calculate the covariance. So lets quickly define it.

# %%
def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n

# %% [markdown]
# The second pre-processing method is called **whitening**. The goal here is to linearly transform the observed signals X in a way that potential correlations between the signals are removed and their variances equal unity. As a result the covariance matrix of the whitened signals will be equal to the identity matrix


def whiten(x):
     # Calculate the covariance matrix
     coVarM = covariance(x) 
   
     # Single value decoposition
     U, S, V = np.linalg.svd(coVarM)
   
     # Calculate diagonal matrix of eigenvalues
     d = np.diag(1.0 / np.sqrt(S)) 
   
     # Calculate whitening matrix
     whiteM = np.dot(U, np.dot(d, U.T))
  
     # Project onto whitening matrix
     Xw = np.dot(whiteM, x) 
   
     return Xw, whiteM


# Calculate Kurtosis

def kurt(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3

    return kurt, skew, var, mean
# %%
