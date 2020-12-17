import numpy as np
# import numba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import FastICA, PCA
import neurokit2
import time

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

import pandas as pd




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


def monte_carlo_run(n_samples,ica_method,seed = None):
    """
    Do a Monte Carlo simulation of n_samples, plot a histogram, mean and standard deviation
    and write results to csv
    :param n_samples: number of runs
    :param measure(): The evalutation measure has to be method
    :param matrix1: np.array of first parameters used in method,
    :param matrix12: np.array of second parameters used in method

    :return: 
    """
    methods = ["jade","power_ica","fast_ica","radical"]
    
    assert (ica_method in methods), \
        "Can't choose  '%s' as ICA - method, possible optiions: %s" %(ica_method, methods)
  
    
    # create example signals:
    data = np.stack([create_signal(c='ecg'),
            create_signal(ampl=4,c='cos'),
            create_signal(c='rect'),
            create_signal(c='sawt')]).T

    # create mixing matrix and mixed signals
    c, r = data.shape

    data_storage = np.empty(n_samples)

    new_MM = True
    
    #time tracking
    t0= time.time()

    #start Monte-Carlo run
    for i in range(n_samples):
        
        if(new_MM):
            MM = mixing_matrix(r,seed)
            mixdata = MM@data.T
            #apply noise
            noise_lvl = 20
            mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=noise_lvl),prop=0.001,std=5) for dat in mixdata])
            # centering the data and whitening the data:
            white_data,W_whiten,W_dewhiten = whitening(mixdata_noise, type='sample')
            if(seed is not None):
                new_MM = False

        if(ica_method == "jade"):
            # Perform JADE
            W = jadeR(white_data,is_whitened=True, verbose = False)
            W = np.squeeze(np.asarray(W))
        elif (ica_method == "power_ica"):
            # Perform PowerICA
            W, _ = PowerICA.powerICA(white_data, 'pow3')
        elif (ica_method == "fast_ica"):
            # Perform FastICA
            ica = FastICA(n_components=4)
            S_ = ica.fit_transform(white_data)  # Get the estimated sources
            W = ica._unmixing
        elif (ica_method == "radical"):
            # Perform Radical
            W = RADICAL(white_data)
        else:
            return
        
        data_storage[i] = md(W_whiten @ MM,W)
        print(data_storage[i])

    # elapsed time for Monte-Carlo Run
    t1 = time.time() - t0
    print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)   
        
    mu = np.mean(data_storage)
    sigma = np.std(data_storage)

    plt.figure()
    plt.hist(data_storage,density = False )  # `density=False` would make counts
    plt.ylabel('count')
    plt.xlabel('Minimum Distance: %s ' % ica_method)
    plt.show()

    # plt.figure()
    # sns.distplot(data_storage[0,:])

    print('Mean %s: %f ' % (ica_method,mu))
    print('Standard deviation %s: %f' %(ica_method,sigma))


    mc_data = { 'Method': [ica_method], '# Runs' : [n_samples], 'Noise-level': [noise_lvl], 'Seed': [seed], 'Mean': [mu], 'Std': [sigma], 'Time elapsed' : [t1]}
    df = pd.DataFrame(mc_data, columns= ['Method', '# Runs','Noise-level', 'Seed','Mean','Std','Time elapsed'])
    df.to_csv(os.path.join(BASE_DIR,'utils','Monte_Carlo_runs.csv'), index = False, header=True, mode = 'a')



if __name__ == "__main__":

    # do a monte-carlo run
    monte_carlo_run(100,'jade', seed = None)



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