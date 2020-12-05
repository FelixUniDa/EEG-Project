#%%
import numpy as np

# %%
def center(x):
    #mean = np.mean(x, axis=1, keepdims=True)
    mean = np.apply_along_axis(np.mean,axis=1,arr=x)
    centered =  x
    centered[0,:] = centered[0,:]-mean[0]
    centered[1,:] = centered[1,:]-mean[1]
    centered[2,:] = centered[2,:]-mean[2]
    centered[3,:] = centered[3,:]-mean[3]
    #print(centered)
    return centered, mean

# %% [markdown]
# For the second pre-processing technique we need to calculate the covariance. So lets quickly define it.

# %%
def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m**2)/n

# %% [markdown]
# The second pre-processing method is called **whitening**. The goal here is to linearly transform the observed signals X in a way that potential correlations between the signals are removed and their variances equal unity. As a result the covariance matrix of the whitened signals will be equal to the identity matrix


def whiten(x):
    m,n = np.shape(x)

    cov = np.cov(x,bias=True)

    d, E = np.linalg.eigh(cov)

    print(d,E)
    D_inv = np.diag(1/np.sqrt(d))

    W_whiten =  D_inv @ E.T
    print(W_whiten)
    x_whiten = (W_whiten @ x)

    return x_whiten
    


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
