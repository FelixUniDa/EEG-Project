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
from utils import mixing_matrix, create_signal, create_outlier, apply_noise, whitening, SNR, MSE, md

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


def monte_carlo_run(n_runs, data_size, ica_method, seed=None, noise_lvl = 20, p_outlier = 0.0, outlier_type='impulse', partition = 30):
    """
    Do a Monte Carlo simulation of n_runs, plot a histogram, mean and standard deviation
    and write results to csv
    :param n_runs: number of runs
    :param measure(): The evalutation measure has to be method
    :param matrix1: np.array of first parameters used in method,
    :param matrix12: np.array of second parameters used in method

    :return: 
    """
    methods = ["jade", "power_ica", "fast_ica", "radical", "coro_ica", "random"]
    
    assert (ica_method in methods), "Can't choose  '%s' as ICA - method, possible optiions: %s" %(ica_method, methods)

    # create example signals:
    data = np.stack([create_signal(x=data_size, c='ecg'),
            create_signal(x=data_size, ampl=1, c='cos'),
            create_signal(x=data_size, c='rect'),
            create_signal(x=data_size, c='sawt')]).T

    # create mixing matrix and mixed signals
    c, r = data.shape

    #data_storage = np.empty(n_runs)

    md_data = np.empty(n_runs)
    SNR_data = np.empty(n_runs)
    MSE_data =np.empty(n_runs)

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

            noise = mixdata_noise - mixdata
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
        elif (ica_method == "random"):
            # Perform Radical
            W = np.random.rand(r,r)
        else:
            return
        
        md_data[i] = md(W_whiten @ MM, W)
        MSE_data[i] = np.mean(MSE(W @ white_data-W @ W_whiten @ noise, data.T))
        SNR_data[i] = np.mean(SNR(W @ white_data-W @ W_whiten @ noise, data.T))
       
        #data_storage[i] = md(W_whiten @ MM, W)
        #print(data_storage[i])

    # elapsed time for Monte-Carlo Run
    t1 = time.time() - t0
    print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)   

    md_mu = np.mean(md_data)
    md_sigma = np.std(md_data)
    md_med = np.median(md_data)
    md_nMAD = scipy.stats.median_abs_deviation(md_data, scale=1/1.4826)

    MSE_mu = np.mean(MSE_data)
    MSE_sigma = np.std(MSE_data)
    MSE_med = np.median(MSE_data)
    MSE_nMAD = scipy.stats.median_abs_deviation(MSE_data, scale=1/1.4826)

    SNR_mu = np.mean(SNR_data)
    SNR_sigma = np.std(SNR_data)
    SNR_med = np.median(SNR_data)
    SNR_nMAD = scipy.stats.median_abs_deviation(SNR_data, scale=1/1.4826)
       
    # mu = np.mean(data_storage)
    # sigma = np.std(data_storage)
    # med = np.median(data_storage)
    # nMAD = scipy.stats.median_abs_deviation(data_storage, scale=1/1.4826)

    # # FRESH SEABORN PLOT - Histogram 
    # x = pd.Series(data_storage, name=('Minimum Distance: %s ' % ica_method))
    # sns.displot(x, kde=True)
    # plt.ylabel('count')
    # file_name = ica_method + '_' + str(n_runs) + '_' + str(noise_lvl) +'dB_' + '.jpg'
    # plt.savefig(os.path.join('results_Monte_Carlo',file_name), dpi=300)  
    # plt.show()

    print('Mean MD %s: %f ' % (ica_method, md_mu))
    print('Standard deviation MD %s: %f' %(ica_method, md_sigma))
    print('Mean MSE %s: %f ' % (ica_method, MSE_mu))
    print('Standard deviation MSE %s: %f' %(ica_method, MSE_sigma))
    print('Mean SNR %s: %f ' % (ica_method, SNR_mu))
    print('Standard deviation SNR %s: %f' %(ica_method, SNR_sigma))

    # print('Mean %s: %f ' % (ica_method, mu))
    # print('Standard deviation %s: %f' %(ica_method, sigma))

    if seed == None:
        seed = 'None'

    mc_data = {'Method': [ica_method], '# Runs': [n_runs], 'Sample Size': [data_size], 'Noise-level [dB]': [noise_lvl],'Percentage Outliers (3Std)': [p], 'Seed': [seed], 'Mean_MD': [md_mu], 'Std_MD': [md_sigma], 'Median_MD': [md_med], 'nMAD_MD': [md_nMAD],'Mean_MSE': [MSE_mu], 'Std_MSE': [MSE_sigma], 'Median_MSE': [MSE_med], 'nMAD_MSE': [MSE_nMAD],'Mean_SNR': [SNR_mu], 'Std_SNR': [SNR_sigma], 'Median_SNR': [SNR_med], 'nMAD_SNR': [SNR_nMAD], 'Time elapsed [s]': [t1]}
    df = pd.DataFrame(mc_data, columns=['Method', '# Runs', 'Sample Size', 'Noise-level [dB]', 'Percentage Outliers (3Std)', 'Seed', 'Mean_MD', 'Std_MD', 'Median_MD', 'nMAD_MD', 'Mean_MSE', 'Std_MSE', 'Median_MSE', 'nMAD_MSE', 'Mean_SNR', 'Std_SNR', 'Median_SNR', 'nMAD_SNR', 'Time elapsed [s]'])
    df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_PowerICA', 'Monte_Carlo_runs_PowerICA.csv'), index=True, header=True, mode='a')

    # mc_data = {'Method': [ica_method], '# Runs': [n_runs], 'Sample Size': [data_size], 'SNR [dB]': [noise_lvl], 'Percentage Outliers (3Std)': [p], 'Seed': [seed], 'Mean': [mu], 'Std': [sigma], 'Median': [med], 'nMAD': [nMAD], 'Time elapsed [s]': [t1]}
    # df = pd.DataFrame(mc_data, columns=['Method', '# Runs', 'Sample Size', 'SNR [dB]', 'Percentage Outliers (3Std)', 'Seed', 'Mean', 'Std', 'Median', 'nMAD', 'Time elapsed [s]'])
    # df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_CoroICA', 'Monte_Carlo_runs_CoroICA.csv'), index=True, header=True, mode='a')

    # return minimum distances stored in data_storage
    return  md_data,MSE_data,SNR_data #data_storage


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
    # df = pd.DataFrame()
    df_md = pd.DataFrame()
    df_MSE = pd.DataFrame()
    df_SNR = pd.DataFrame()

    ica_method = 'coro_ica'  # further changes need to be made in plt.savefig & !df.to_csv in def monte_carlo!
    folder_to_save = 'results_Monte_Carlo_CoroICA'
    type_list_to_test = ["Type 1","Type 5"]  # "Type 1", "Type 2", "Type 3", "Type 4"
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
                # mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type)

                # d = {'Minimum Distance': mds, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)

                mds,MSEs,SNRs = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type)

                d_SNR = {'Signal to Noise Ratio (Mixed Signals) in dB': SNRs, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': MSEs, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_md = {'Minimum Distance': mds, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp_md = pd.DataFrame(data=d_md)
                temp_MSE = pd.DataFrame(data=d_MSE)
                temp_SNR = pd.DataFrame(data=d_SNR)
                df_md = df_md.append(temp_md)
                df_MSE = df_MSE.append(temp_MSE)
                df_SNR = df_SNR.append(temp_SNR)

                print("Ready samplesize {}".format(s))

            # title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
            #     noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            # file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
            #     noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            # sns.boxplot(x='Sample Size', y='Minimum Distance', data=df).set_title(title)
            # plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            # #plt.show()
            plt.clf()
            plt.ylim(top=1,bottom=0)
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            title =  ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = 'MD_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
        
            # fig, axes = plt.subplots(3, 1, figsize=(10, 18))
            # fig.suptitle('Metrics')
            sns.boxplot(ax = axes[0], x='Sample Size', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()
            

            title_MSE = 'MSE_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax = axes[1],x='Sample Size', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       

            plt.ylim(top=60,bottom=0)
            title_SNR = 'SNR_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax = axes[2],x='Sample Size', y='Signal to Noise Ratio (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            #plt.savefig(os.path.join(folder_to_save, file_name_SNR), dpi=300)
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            plt.show()

        if name == "Type 5":
            outlier_type = "impulse"
            for snr in noise:
                # do a monte-carlo run
                s = sample_size
                # mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=snr, p_outlier=p)

                # d = {'Minimum Distance': mds, 'SNR': np.repeat(snr, n_runs),
                #      '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)

                mds,MSEs,SNRs = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=snr, p_outlier=p, outlier_type=outlier_type)

                d_SNR = {'Signal to Noise Ratio (Mixed Signals) in dB': SNRs, 'SNR(additive noise)': np.repeat(snr, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': MSEs, 'SNR(additive noise)': np.repeat(snr, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_md = {'Minimum Distance': mds, 'SNR(additive noise)': np.repeat(snr, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp_md = pd.DataFrame(data=d_md)
                temp_MSE = pd.DataFrame(data=d_MSE)
                temp_SNR = pd.DataFrame(data=d_SNR)
                df_md = df_md.append(temp_md)
                df_MSE = df_MSE.append(temp_MSE)
                df_SNR = df_SNR.append(temp_SNR)

                print("Ready noise level {} with sample size {}".format(snr, s))

            # title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
            #     sample_size) + ', ' + str(p*10) + ' % outliers'
            # file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
            #     noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            # sns.boxplot(x='SNR', y='Minimum Distance', data=df).set_title(title)
            # plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            #plt.show()
            plt.clf()
            plt.ylim(top=1,bottom=0)
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = 'MD_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
        
            # fig, axes = plt.subplots(3, 1, figsize=(10, 18))
            # fig.suptitle('Metrics')
            sns.boxplot(ax = axes[0],x='SNR(additive noise)', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = 'MSE_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax = axes[1],x='SNR(additive noise)', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       

            plt.ylim(top=60,bottom=0)
            title_SNR = 'SNR_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax = axes[2],x='SNR(additive noise)', y='Signal to Noise Ratio (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            plt.show()

        if name == "Type 6" or name == "Type 7":
            outlier_type = type_dict.get(name)[4]
            for percentage in p:
                # do a monte-carlo run
                s = sample_size
                # mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=percentage, )

                # d = {'Minimum Distance': mds, 'Outlier Percentage': np.repeat(percentage, n_runs),
                #      '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)

                mds,MSEs,SNRs = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=percentage, outlier_type=outlier_type)

                d_SNR = {'Signal to Noise Ratio (Mixed Signals) in dB': SNRs, 'Outlier Percentage': np.repeat(percentage, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': MSEs, 'Outlier Percentage': np.repeat(percentage, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_md = {'Minimum Distance': mds, 'Outlier Percentage': np.repeat(percentage, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp_md = pd.DataFrame(data=d_md)
                temp_MSE = pd.DataFrame(data=d_MSE)
                temp_SNR = pd.DataFrame(data=d_SNR)
                df_md = df_md.append(temp_md)
                df_MSE = df_MSE.append(temp_MSE)
                df_SNR = df_SNR.append(temp_SNR)

                print("Ready outlier percentage {} with sample size {}".format(percentage, s))

            # title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
            #     sample_size) + ', ' + str(
            #     noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            # file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
            #     noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            # sns.boxplot(x='Outlier Percentage', y='Minimum Distance', data=df).set_title(title)
            # plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)
            #plt.show()

            plt.clf()
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = 'MD_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
           
            sns.boxplot(ax = axes[0], x='Outlier Percentage', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = 'MSE_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            sns.boxplot(ax = axes[1], x='Outlier Percentage', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       


            title_SNR = 'SNR_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            sns.boxplot(ax = axes[2],x='Outlier Percentage', y='Signal to Noise Ratio (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            plt.savefig(os.path.join(folder_to_save, file_name_SNR), dpi=300)
            plt.show()

        if name == "Type 8" :
            outlier_type = "impulse"
            partitions = np.arange(50,400,10) #np.ones(41)/np.arange(50,9,-1)*sample_size 
            for ps in partitions:
                # do a monte-carlo run
                #mds = monte_carlo_run(n_runs, sample_size , ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))

                # d = {'Minimum Distance': mds, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)
                mds,MSEs,SNRs = monte_carlo_run(n_runs, sample_size, ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))


                d_SNR = {'Signal to Noise Ratio (Mixed Signals) in dB': SNRs, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': MSEs, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_md = {'Minimum Distance': mds, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                temp_md = pd.DataFrame(data=d_md)
                temp_MSE = pd.DataFrame(data=d_MSE)
                temp_SNR = pd.DataFrame(data=d_SNR)
                df_md = df_md.append(temp_md)
                df_MSE = df_MSE.append(temp_MSE)
                df_SNR = df_SNR.append(temp_SNR)

                print("Ready Partitionsize {}".format(ps))

            # title = ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
            #     noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            # file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
            #     noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            # sns.boxplot(x='Partition Size', y='Minimum Distance', data=df).set_title(title)
            # plt.show()
            # plt.savefig(os.path.join(folder_to_save, file_name), dpi=300)

            plt.clf()
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = 'MD_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
           
            sns.boxplot(ax = axes[0], x='Partition Size', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = 'MSE_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'

            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            sns.boxplot(ax = axes[1], x='Partition Size', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       


            title_SNR = 'SNR_' + ica_method + ',  ' + name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            sns.boxplot(ax = axes[2], x='Partition Size', y='Signal to Noise Ratio (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            plt.savefig(os.path.join(folder_to_save, file_name_SNR), dpi=300)
            plt.show()

    print("--- Success after %s seconds ---" % (time.time() - start_time))
