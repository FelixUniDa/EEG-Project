import numpy as np
# import numba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import FastICA, PCA
import neurokit2
import time
from scipy.stats import median_abs_deviation
import scipy

import os
import sys

#from Python_Code.Compare_ICA_algos.fast_Radical import RADICAL
from utils import mixing_matrix, create_signal, create_outlier, apply_noise, whitening

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath( __file__ )))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'JADE'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'Compare_ICA_algos'))

import PowerICA
from fast_Radical import *
from jade import jadeR
# from distances import *
import pandas as pd
from coroica import CoroICA

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


def monte_carlo_run(n_runs, data_size, ica_method, seed=None, noise_lvl = 20, p_outlier = 0.0):
    """
    Do a Monte Carlo simulation of n_runs, plot a histogram, mean and standard deviation
    and write results to csv
    :param n_runs: number of runs
    :param measure(): The evalutation measure has to be method
    :param matrix1: np.array of first parameters used in method,
    :param matrix12: np.array of second parameters used in method

    :return: 
    """
    methods = ["jade", "power_ica", "fast_ica", "radical", "coro_ica"]
    
    assert (ica_method in methods), "Can't choose  '%s' as ICA - method, possible optiions: %s" %(ica_method, methods)

    # create example signals:
    data = np.stack([create_signal(x=data_size, c='ecg'),
            create_signal(x=data_size, ampl=1, c='cos'),
            create_signal(x=data_size, c='rect'),
            create_signal(x=data_size, c='sawt')]).T

    # create mixing matrix and mixed signals
    c, r = data.shape

    data_storage = np.empty(n_runs)

    new_MM = True
    
    #time tracking
    t0 = time.time()

    #start Monte-Carlo run
    for i in range(n_runs):
        
        if(new_MM):
            MM = mixing_matrix(r, None)
            #print(MM)
            mixdata = MM@data.T
            #apply noise and/or create outlier
            mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=noise_lvl),prop=p_outlier,std=3) for dat in mixdata])

            # centering the data and whitening the data:
            white_data, W_whiten, W_dewhiten,_ = whitening(mixdata_noise, type='sample')
            if(seed is not None):
                new_MM = False

        # print(mixdata_noise.shape)
        # print(white_data.shape)
        if (ica_method == "jade"):
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
        elif (ica_method == 'coro_ica'):
            #print(r)
            c = CoroICA(partitionsize=100, groupsize=10000)
            c.fit(white_data.T)
            W = c.V_
        else:
            return
        
        data_storage[i] = md(W_whiten @ MM,W)
        #print(data_storage[i])

    # elapsed time for Monte-Carlo Run
    t1 = time.time() - t0
    print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)   
        
    mu = np.mean(data_storage)
    sigma = np.std(data_storage)
    med = np.median(data_storage)
    nMAD = scipy.stats.median_abs_deviation(data_storage, scale=1/1.4826)

    # # FRESH SEABORN PLOT - Histogram 
    # x = pd.Series(data_storage, name=('Minimum Distance: %s ' % ica_method))
    # sns.displot(x, kde=True)
    # plt.ylabel('count')
    # file_name = ica_method + '_' + str(n_runs) + '_' + str(noise_lvl) +'dB_' + '.jpg'
    # plt.savefig(os.path.join('results_Monte_Carlo',file_name), dpi=300)  
    # plt.show()

    print('Mean %s: %f ' % (ica_method, mu))
    print('Standard deviation %s: %f' %(ica_method, sigma))

    if seed == None:
        seed = 'None'



    mc_data = {'Method': [ica_method], '# Runs' : [n_runs], 'Sample Size' : [data_size], 'Noise-level [dB]': [noise_lvl],'Percentage Outliers (3Std)': [p], 'Seed': [seed], 'Mean': [mu], 'Std': [sigma], 'Median': [med],'nMAD': [nMAD], 'Time elapsed [s]' : [t1]}
    df = pd.DataFrame(mc_data, columns= ['Method', '# Runs', 'Sample Size', 'Noise-level [dB]', 'Percentage Outliers (3Std)', 'Seed', 'Mean', 'Std', 'Median', 'nMAD', 'Time elapsed [s]'])
    df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_JADE', 'Monte_Carlo_runs_JADE.csv'), index=True, header=True, mode='a')

    # return minimum distances stored in data_storage
    return data_storage


if __name__ == "__main__":
    # track time for how long this old endures
    start_time = time.time()
    # steps of sample_size
    #steps = 
    n_runs = 10000
    sample_size = np.array([1000, 2500, 5000, 10000, 15000]) #1000, 2500, 5000, 10000, 15000
    # init DataFrame
    df = pd.DataFrame()
    ica_method = 'jade'
    noise = 40
    p = 0.0
    for s in sample_size:
        # do a monte-carlo run
        mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=p)
        
        d = {'Minimum Distance': mds, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
        temp = pd.DataFrame(data=d)
        df = df.append(temp)
        print("Ready samplesize %s" %s)

    title = ica_method + ',  ' + 'runs: ' + str(n_runs) + ', ' + str(noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
    file_name = ica_method + '_' + str(n_runs) + 'Runs' + '_' + str(noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
    sns.boxplot(x='Sample Size', y='Minimum Distance', data=df).set_title(title)
    plt.savefig(os.path.join('results_Monte_Carlo_JADE', file_name), dpi=300)
    # plt.show()
    print("--- Success after %s seconds ---" % (time.time() - start_time))

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