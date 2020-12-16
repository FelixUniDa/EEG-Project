import numpy as np
# import numba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import FastICA, PCA
import neurokit2

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath( __file__ )))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'Python_Code','JADE'))
sys.path.append(os.path.join(BASE_DIR,'utils'))
sys.path.append(os.path.join(BASE_DIR,'Python_Code','Compare_ICA_algos'))

import PowerICA
from fast_Radical import *
from jade import jadeR
from distances import *


""" Monte Carlo Simulation Experminets"""
# inspired by:
# (1) http://justinbois.github.io/bootcamp/2015/lessons/l24_monte_carlo.html
# (2) https://seaborn.pydata.org/generated/seaborn.distplot.html
# (3)
# seaborn settings
sns.set_theme(style="darkgrid")


# ToDo: rewrite Skeleton for our needs
# ToDo: Time Tracking
# ToDo: Create Pandas DataFrame that read/saves the data
# ToDo: write experiment results to csv


def monte_carlo_run(n_samples,ica_method,mixing_mat,W_whiten,signals):
    """
    Do a Monte Carlo simulation of n_samples, plot a histogram, mean and standard deviation
    and write results to csv
    :param n_samples: number of runs
    :param measure(): The evalutation measure has to be method
    :param matrix1: np.array of first parameters used in method,
    :param matrix12: np.array of second parameters used in method

    :return: 
    """
    # assert callable(ica_method), \
    #     " The given measure is not callable"
  
    # assert (isinstance(mixing_mat, type(np.array) ) & isinstance(signals, type(np.ndarray))),\
    #     "A is of type (%s) and B is of type (%s), should be (%s)" % (type(mixing_mat),type(signals), "np.array")
    

    data_storage = np.empty((1,n_samples))

    for i in range(n_samples):
        # TODO: assert if ica_method in names
        if(ica_method == "jade"):
            # perform Jade
            W = jadeR(white_data,is_whitened=True, verbose = False)
            W = np.squeeze(np.asarray(W))
        elif (ica_method == "power_ica"):
            # perform PowerICA
            W, _ = PowerICA.powerICA(white_data, 'pow3')
        elif (ica_method == "fast_ica"):
            # compute ICA
            ica = FastICA(n_components=4)
            S_ = ica.fit_transform(white_data)  # Get the estimated sources
            W = ica._unmixing
        elif (ica_method == "radical"):
            #Perform fastICA
            W = RADICAL(white_data)
        else:
            return
        #print(i)
        data_storage[0,i] = md(W_whiten @ mixing_mat,W)
        #print(data_storage[0,i])
        

    mu = float(np.mean(data_storage,axis = 1))
    sigma = np.std(data_storage, axis = 1)

    plt.figure()
    #sns.distplot(data_storage[1,:])
    plt.hist(data_storage[0], density=True)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Minimum Distance: %s ' % ica_method)
    print('Mean %s: %f ' % (ica_method,mu))
    print('Standard deviation %s: %f' %(ica_method,sigma))
    



# def random_walk_1d(n_steps):
#     """
#     take a random walk of n_steps and return excursion from the origin
#     :param n_steps: number of stebern
#     :return: sum of steps
#     """

#     # generate an array of our steps (+1 for right, -1 for left)
#     steps = 2 * np.random.randint(0, 2, n_steps) - 1
#     # sum of steps
#     return steps.sum()

# # #steps to take for computing distribution
# n_steps = 10000

# # number of random walks to take
# n_samples = 10000

# # initial samples of displacements
# x = np.empty(n_samples)

# # take all of the random walks
# for i in range(n_samples):
#     x[i] = random_walk_1d(n_steps)

# # make histogram and boxplot visualization for result
# ax = sns.distplot(x)


# # ax2.set(ylim=(-.5, 10))

# x = pd.Series(x, name='$x$ (a.u.)')
# sns.displot(x, kde=True)
# plt.show()


# # trial prints
# print(' Mean displacement:', x.mean())
# print('Standard deviation:', x.std())


if __name__ == "__main__":
    # create example signals:
    data = np.stack([create_signal(c='ecg'),
            create_signal(ampl=4,c='cos'),
            create_signal(c='rect'),
            create_signal(c='sawt')]).T

    # create mixing matrix and mixed signals
    c, r = data.shape
    MM = mixing_matrix(r,seed=1)
    mixdata = MM@data.T

    #apply noise
    mixdata_noise = np.stack([create_outlier(apply_noise(dat,c='white', SNR_dB=20),prop=0.001,std=5) for dat in mixdata])

    # centering the data and whitening the data:
    white_data,W_whiten,W_dewhiten = whitening(mixdata_noise, type='sample')
    
    monte_carlo_run(10,'radical',MM,W_whiten,mixdata_noise)



