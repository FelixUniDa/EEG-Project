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


def monte_carlo_run(n_runs, data_size, ica_method, seed=None, noise_lvl = 20, p_outlier = 0.0, outlier_type='impulse', partition = 100):
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
            mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=noise_lvl),prop=p_outlier,std=3,type = 'impulse') for dat in mixdata])
            mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl), prop=p_outlier, std=3, type=outlier_type) for dat in mixdata])

            # centering the data and whitening the data:
            white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample')
            if(seed is not None):
                new_MM = False

        # print(mixdata_noise.shape)
        # print(white_data.shape)
        if (ica_method == "jade"):
            # Perform JADE
            W = jadeR(white_data, is_whitened=True, verbose=False)
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
            c = CoroICA(partitionsize = partition,groupsize= data_size)
            c.fit(white_data.T)
            W = c.V_
        else:
            return
        
        data_storage[i] = md(W_whiten @ MM, W)
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

    mc_data = {'Method': [ica_method], '# Runs': [n_runs], 'Sample Size': [data_size], 'SNR [dB]': [noise_lvl], 'Percentage Outliers (3Std)': [p], 'Seed': [seed], 'Mean': [mu], 'Std': [sigma], 'Median': [med], 'nMAD': [nMAD], 'Time elapsed [s]': [t1]}
    df = pd.DataFrame(mc_data, columns=['Method', '# Runs', 'Sample Size', 'SNR [dB]', 'Percentage Outliers (3Std)', 'Seed', 'Mean', 'Std', 'Median', 'nMAD', 'Time elapsed [s]'])
    df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_CoroICA', 'Monte_Carlo_runs_CoroICA.csv'), index=True, header=True, mode='a')

    # return minimum distances stored in data_storage
    return data_storage


if __name__ == "__main__":

    # def of types to test
    sample_size_full = np.array([1000, 2500, 5000, 10000, 15000])  # 1000, 2500, 5000, 10000, 15000
    noise_list = np.array([40, 30, 20, 10, 6, 3])
    outlier_list = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.10, 0.20, 0.50])
    n_runs = 100
    type_dict = dict()
    type_dict.update({"Type 1": [1000, 0, n_runs, sample_size_full]})  # no noise, no outlier, runs, samples
    type_dict.update({"Type 2": [40, 0, n_runs, sample_size_full]})  # 40db noise, no outlier, runs, samples
    type_dict.update({"Type 3": [1000, 0.001, n_runs, sample_size_full]})  # no noise, 0.1 % outlier, runs, samples
    type_dict.update({"Type 4": [40, 0.001, n_runs, sample_size_full]})  # 40 db noise, 0.1 % outlier, runs, samples
    type_dict.update({"Type 5": [noise_list, 0, n_runs, 10000]})
    type_dict.update({"Type 6": [1000, outlier_list, n_runs, 10000, 'patch']})
    type_dict.update({"Type 7": [1000, outlier_list, n_runs, 10000, 'impulse']})
    # Coro_ica partitionsize test. 
    type_dict.update({"Type 8": [40, 0.000, n_runs, 2500]}) #(dB Noise, outlier proportion, runs, sample_size, partions_list)


    # track time for how long this run takes
    start_time = time.time()

    # init DataFrame
    df = pd.DataFrame()
    ica_method = 'coro_ica'  # further changes need to be made in plt.savefig & !df.to_csv in def monte_carlo!
    folder_to_save = 'results_Monte_Carlo_CoroICA'
    type_list_to_test = ["Type 8"]  # "Type 1", "Type 2", "Type 3", "Type 4"
    for name in type_list_to_test:
        # parameter for each type to test
        noise = type_dict.get(name)[0]
        p = type_dict.get(name)[1]
        n_runs = type_dict.get(name)[2]
        sample_size = type_dict.get(name)[3]

        if name == "Type 1" or name == "Type 2" or name == "Type 3" or name == "Type 4":
            outlier_type = "impulse"
            for s in sample_size:
                # do a monte-carlo run
                mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type)

                d = {'Minimum Distance': mds, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp = pd.DataFrame(data=d)
                df = df.append(temp)
                print("Ready samplesize {}".format(s))

            title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(x='Sample Size', y='Minimum Distance', data=df).set_title(title)
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            #plt.show()

        if name == "Type 5":
            outlier_type = "impulse"
            for snr in noise:
                # do a monte-carlo run
                s = sample_size
                mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=snr, p_outlier=p)

                d = {'Minimum Distance': mds, 'SNR': np.repeat(snr, n_runs),
                     '# MC Runs': np.repeat(n_runs, n_runs)}
                temp = pd.DataFrame(data=d)
                df = df.append(temp)
                print("Ready noise level {} with sample size {}".format(snr, s))

            title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(p*10) + ' % outliers'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(x='SNR', y='Minimum Distance', data=df).set_title(title)
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            #plt.show()

        if name == "Type 6" or name == "Type 7":
            outlier_type = type_dict.get(name)[4]
            for percentage in p:
                # do a monte-carlo run
                s = sample_size
                mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=percentage, )

                d = {'Minimum Distance': mds, 'Outlier Percentage': np.repeat(percentage, n_runs),
                     '# MC Runs': np.repeat(n_runs, n_runs)}
                temp = pd.DataFrame(data=d)
                df = df.append(temp)
                print("Ready outlier percentage {} with sample size {}".format(percentage, s))

            title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            sns.boxplot(x='Outlier Percentage', y='Minimum Distance', data=df).set_title(title)
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            #plt.show()

        if name == "Type 8" :
            outlier_type = "impulse"
            partitions = np.arange(50,400,10) #np.ones(41)/np.arange(50,9,-1)*sample_size 
            for ps in partitions:
                # do a monte-carlo run
                mds = monte_carlo_run(n_runs, sample_size , ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))

                d = {'Minimum Distance': mds, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp = pd.DataFrame(data=d)
                df = df.append(temp)
                print("Ready Partitionsize {}".format(ps))

            title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            sns.boxplot(x='Partition Size', y='Minimum Distance', data=df).set_title(title)
            plt.show()
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
    print("--- Success after %s seconds ---" % (time.time() - start_time))

