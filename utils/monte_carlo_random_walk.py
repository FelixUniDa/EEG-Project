import numpy as np
# import numba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import FastICA, PCA
import neurokit2
import time
from scipy.stats import hmean,median_abs_deviation
import scipy

import os
import sys

#from Python_Code.Compare_ICA_algos.fast_Radical import RADICAL
from utils_neu import mixing_matrix, create_signal, create_outlier, apply_noise, whitening, SNR, MSE, md, add_artifact

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
sns.set_context("poster")


# ToDo: rewrite Skeleton for our needs
# ToDo: Time Tracking
# ToDo: Create Pandas DataFrame that read/saves the data
# ToDo: write experiment results to csv


def monte_carlo_run(n_runs, data_size, ica_method, data_type='standard', seed=None, noise_lvl=20, std_outlier=3, p_outlier=0.0, outlier_type='impulse', partition=30):
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
    data_types = ['standard', 'delorme']

    assert (ica_method in methods), "Can't choose  '%s' as ICA - method, possible options: %s" %(ica_method, methods)
    assert (data_type in data_types), "Can't choose  '%s' as datatype, possible options: %s" % (data_type, data_types)

    # create example signals: 
    # choose Frequencies in the range of brainwaves (Alpha to Delta)
    # DELTA WAVES (0.5 - 3 HZ) 
    # THETA WAVES (3 - 8 HZ)
    # ALPHA WAVES (8 - 12 HZ)
    # BETA WAVES (12 - 38 HZ)
    # LOW GAMMA WAVES (38 - 70 HZ)
    # HIGH GAMMA WAVES (70 - 150 HZ )
    # https://en.wikipedia.org/wiki/Neural_oscillation
    # https://brainworksneurotherapy.com/what-are-brainwaves

    data = np.stack([create_signal(f=2, x=data_size, c='ecg'),
            create_signal(f=5, x=data_size, ampl=1, c='cos'),
            create_signal(f=10, x=data_size, c='rect'),
            create_signal(f=25, x=data_size, c='sawt')]).T

    # create mixing matrix and mixed signals
    c, r = data.shape

    # data_storage = np.empty(n_runs)

    md_data = np.zeros(n_runs)
    SNR_data = np.zeros(n_runs)
    SNR_mean_data = np.zeros(n_runs)
    SNR_med_data = np.zeros(n_runs)
    SNR_hmean_data = np.zeros(n_runs)
    MSE_data = np.zeros(n_runs)
    MSE_mean_data = np.zeros(n_runs)
    MSE_med_data = np.zeros(n_runs)

    new_MM = True
    
    # time tracking
    t0 = time.time()

    # start Monte-Carlo run
    for i in range(n_runs):
        
        if(new_MM):

            if data_type == 'standard':
                MM = mixing_matrix(r, None)
                # print(MM)
                mixdata = MM @ data.T
                # apply noise and/or create outlier
                mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl), std=std_outlier, prop=p_outlier, type=outlier_type) for dat in mixdata])
                noise = mixdata_noise - mixdata
                # centering the data and whitening the data:
                white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample')

            if data_type == 'delorme':
                # parameter for add_artifact function for delorme artefacts
                fs = 1000
                eeg_components = 4
                eeg_data = data
                delorme_type = 'all'
                prop = 0.1
                snr_dB = 3
                artifacts = 5

                if (delorme_type == 'all'):
                    type = np.array(['eye', 'muscle', 'linear', 'electric', 'noise'])
                    artifacts = 5
                else:
                    type = np.repeat(type, artifacts)

                if (eeg_components < artifacts):
                    artifacts = eeg_components

                eeg_data_artif = np.zeros_like(eeg_data)
                eeg_data_outlier = np.zeros_like(eeg_data)
                for i in range(0, eeg_components):
                    if (i < artifacts):
                        data_outl, outlier = add_artifact(eeg_data[0::, i], fs, prop=prop, snr_dB=snr_dB,
                                                              number=artifacts, type=type[i], seed=None)
                        eeg_data_artif[0::, i] = data_outl
                        eeg_data_outlier[0::, i] = outlier
                    else:
                        eeg_data_artif[0::, i] = eeg_data[0::, i]
                        eeg_data_outlier[0::, i] = eeg_data_outlier[0::, i]

                # plt.plot(eeg_data_artif)
                # plt.show()

                c, r = eeg_data_artif.shape
                MM = mixing_matrix(r, None, m=artifacts)
                # print(MM)
                mixdata = MM @ eeg_data_artif.T
                noise = eeg_data_outlier

                # get
                # noise_lvl = 100
                # p_outlier = 0.0

                # apply noise and/or create outlier
                mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl),
                                                         prop=p_outlier, std=std_outlier, type=outlier_type) for dat in mixdata])

                # centering the data and whitening the data:
                white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample')

                """
                mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl),
                                                         prop=p_outlier, std=3, type='impulse') for dat in mixdata])
                mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl),
                                                         prop=p_outlier, std=3, type=outlier_type) for dat in mixdata])
                noise = mixdata_noise - mixdata
                """

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
            c = CoroICA(partitionsize=partition, groupsize=data_size)
            c.fit(white_data.T)
            W = c.V_
        elif (ica_method == "random"):
            # Perform Radical
            W = np.random.rand(r, r)
        else:
            return
        
        unmixed_data = W @ white_data-W @ W_whiten @ noise
        md_data[i] = md(W_whiten @ MM, W)
        MSE_data[i] = np.mean(MSE(unmixed_data, data.T))
        MSE_mean_data[i] = np.mean(MSE(unmixed_data, data.T))
        MSE_med_data[i] = np.median(MSE(unmixed_data, data.T))
        SNR_data[i] = np.mean(SNR(unmixed_data, data.T))
        SNR_mean_data[i] = np.mean(SNR(unmixed_data, data.T))
        SNR_med_data[i] = np.median(SNR(unmixed_data, data.T))
        SNR_hmean_data[i] = hmean(10**(SNR(unmixed_data, data.T)/10))
       
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
    df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_JADE', 'Monte_Carlo_runs_JADE.csv'), index=True, header=True, mode='a')

    # mc_data = {'Method': [ica_method], '# Runs': [n_runs], 'Sample Size': [data_size], 'SNR [dB]': [noise_lvl], 'Percentage Outliers (3Std)': [p], 'Seed': [seed], 'Mean': [mu], 'Std': [sigma], 'Median': [med], 'nMAD': [nMAD], 'Time elapsed [s]': [t1]}
    # df = pd.DataFrame(mc_data, columns=['Method', '# Runs', 'Sample Size', 'SNR [dB]', 'Percentage Outliers (3Std)', 'Seed', 'Mean', 'Std', 'Median', 'nMAD', 'Time elapsed [s]'])
    # df.to_csv(os.path.join(BASE_DIR, 'utils', 'results_Monte_Carlo_CoroICA', 'Monte_Carlo_runs_CoroICA.csv'), index=True, header=True, mode='a')

    # return minimum distances stored in data_storage
    return md_data, MSE_med_data, SNR_med_data,MSE_data, SNR_data, SNR_hmean_data #data_storage


if __name__ == "__main__":

    # def of types to test
    sample_size_full = np.array([1000, 2500, 5000, 10000, 15000])  # 1000, 2500, 5000, 10000, 15000
    noise_list = np.array([40, 30, 20, 10, 6, 3])
    noise_list2 = np.array([40, 35, 30, 25, 20, 15, 10, 8, 6, 3, 1, 0])
    outlier_list = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.05, 0.10, 0.20, 0.50])
    n_runs = 3
    std_outlier = 100
    type_dict = dict()
    type_dict.update({"Type 1": [1000, 0, n_runs, sample_size_full, 'impulse', std_outlier]})  # no noise, no outlier, runs, samples
    type_dict.update({"Type 2": [40, 0, n_runs, sample_size_full, 'impulse', std_outlier]})  # 40db noise, no outlier, runs, samples
    type_dict.update({"Type 3": [1000, 0.001, n_runs, sample_size_full, 'impulse', std_outlier]})  # no noise, 0.1 % outlier, runs, samples
    type_dict.update({"Type 4": [40, 0.001, n_runs, sample_size_full, 'impulse', std_outlier]})  # 40 db noise, 0.1 % outlier, runs, samples
    type_dict.update({"Type 5": [noise_list, 0, n_runs, 10000, 'impulse', std_outlier]})
    type_dict.update({"Type 6": [1000, outlier_list, n_runs, 10000, 'patch', std_outlier]})
    type_dict.update({"Type 7": [1000, outlier_list, n_runs, 10000, 'impulse', std_outlier]})
    # Coro_ica partitionsize test. 
    type_dict.update({"Type 8": [1000, 0.000, n_runs, 10000, 'impulse', std_outlier]}) # (dB Noise, outlier proportion, runs, sample_size)
    type_dict.update({"Type scat": [noise_list2, 0.000, n_runs, 10000, 'impulse', std_outlier]})

    # check if data should be based on standard 4 signals, delorme artefacts or eeg
    signal_types = ['standard', 'delorme', 'eeg']
    signal_type = signal_types[0]

    # track time for how long this run takes
    start_time = time.time()

    # init DataFrame
    # df = pd.DataFrame()
    df_md = pd.DataFrame()
    df_MSE = pd.DataFrame()
    df_SNR = pd.DataFrame()

    ica_method = 'jade'  # further changes need to be made in plt.savefig & !df.to_csv in def monte_carlo!
    folder_to_save = 'results_Monte_Carlo_JADE'
    type_list_to_test = ["Type scat"]  # "Type 1", "Type 2", "Type 3", "Type 4"

    # quick adjustment -> this is here to make quick checks for the implementations not for the big runs!
    # type_list_to_test = ["Type 6"]

    for name in type_list_to_test:
        # parameter for each type to test
        noise = type_dict.get(name)[0]
        p = type_dict.get(name)[1]
        n_runs = type_dict.get(name)[2]
        sample_size = type_dict.get(name)[3]
        std_outlier = type_dict.get(name)[5]

        if name == "Type 1" or name == "Type 2" or name == "Type 3" or name == "Type 4":
            outlier_type = "impulse"
            for s in sample_size:
                # do a monte-carlo run
                # mds = monte_carlo_run(n_runs, s, ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type)

                # d = {'Minimum Distance': mds, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)

                mds, median_MSEs, median_SNRs,mean_MSEs, mean_SNRs, hSNRs = monte_carlo_run(n_runs, s, ica_method, seed=None, data_type=signal_type, noise_lvl=noise, p_outlier=p, std_outlier=std_outlier, outlier_type=outlier_type)

                d_SNR = {'Signal to Noise Ratio\n (Mixed Signals) in dB': mean_SNRs, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': mean_MSEs, 'Sample Size': np.repeat(s, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
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
            plt.ylim(top=1, bottom=0)
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            plt.subplots_adjust(bottom=0.1, hspace=0.5)
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p*100) + '.jpg'
            fig.suptitle(title)

            title_md = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p*100) + '% outliers'
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
        
            # fig, axes = plt.subplots(3, 1, figsize=(10, 18))
            # fig.suptitle('Metrics')
            sns.boxplot(ax=axes[0], x='Sample Size', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()
            

            title_MSE = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p*100) + '% outliers'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p*100) + '.jpg'
            sns.boxplot(ax = axes[1], x='Sample Size', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       

            plt.ylim(top=60, bottom=0)
            title_SNR = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p*100) + '% outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p*100) + '.jpg'
            sns.boxplot(ax=axes[2], x='Sample Size', y='Signal to Noise Ratio\n (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
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

                mds, median_MSEs, median_SNRs,mean_MSEs, mean_SNRs, hSNRs = monte_carlo_run(n_runs, s, ica_method, seed=None, data_type=signal_type, noise_lvl=snr, p_outlier=p, std_outlier=std_outlier, outlier_type=outlier_type)

                d_SNR = {'Signal to Noise Ratio\n (Mixed Signals) in dB': mean_SNRs, 'SNR(additive noise)': np.repeat(snr, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': mean_MSEs, 'SNR(additive noise)': np.repeat(snr, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
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
            plt.ylim(top=1, bottom=0)
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            plt.subplots_adjust(bottom=0.1, hspace=0.5)
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
        
            # fig, axes = plt.subplots(3, 1, figsize=(10, 18))
            # fig.suptitle('Metrics')
            sns.boxplot(ax=axes[0], x='SNR(additive noise)', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax = axes[1], x='SNR(additive noise)', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       

            plt.ylim(top=60, bottom=0)
            title_SNR = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(noise[-1]) + "-" + str(noise[0]) + 'dB noise' + ', ' + str(p) + '% outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            sns.boxplot(ax=axes[2], x='SNR(additive noise)', y='Signal to Noise Ratio\n (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
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

                mds, median_MSEs, median_SNRs,mean_MSEs, mean_SNRs, hSNRs = monte_carlo_run(n_runs, s, ica_method, data_type=signal_type, seed=None, noise_lvl=noise, p_outlier=percentage, std_outlier=std_outlier, outlier_type=outlier_type)
                percentage = percentage * 100
                d_SNR = {'Signal to Noise Ratio\n (Mixed Signals) in dB': mean_SNRs, 'Outlier Percentage': np.repeat(percentage, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': mean_MSEs, 'Outlier Percentage': np.repeat(percentage, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # adjustment for percentage to be percentage for correct plotting

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
            plt.subplots_adjust(bottom=0.1, hspace=0.5)
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
           
            sns.boxplot(ax = axes[0], x='Outlier Percentage', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            sns.boxplot(ax=axes[1], x='Outlier Percentage', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       


            title_SNR = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + 'sample size:' + str(
                sample_size) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(outlier_type) + ' outlier type'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'type_outlier_' + str(outlier_type) + '.jpg'
            sns.boxplot(ax=axes[2], x='Outlier Percentage', y='Signal to Noise Ratio\n (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            plt.savefig(os.path.join(folder_to_save, file_name_SNR), dpi=300)
            plt.show()

        if name == "Type 8":
            outlier_type = "impulse"
            partitions = np.arange(10, 100, 10) #np.ones(41)/np.arange(50,9,-1)*sample_size
            for ps in partitions:
                # do a monte-carlo run
                #mds = monte_carlo_run(n_runs, sample_size , ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))

                # d = {'Minimum Distance': mds, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)
                mds, median_MSEs, median_SNRs,mean_MSEs, mean_SNRs, hSNRs = monte_carlo_run(n_runs, sample_size, ica_method, data_type=signal_type, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))

                d_SNR = {'Signal to Noise Ratio\n (Mixed Signals) in dB': mean_SNRs, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                d_MSE = {'Mean Squared Error': mean_MSEs, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
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
            plt.subplots_adjust(bottom=0.1, hspace=0.5)
            title = ica_method + ' ' + name + ' Metrics'
            file_name = ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) + '.jpg'
            fig.suptitle(title)

            title_md = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            
            file_name_md = 'MD_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
           
            sns.boxplot(ax=axes[0], x='Partition Size', y='Minimum Distance', data=df_md).set_title(title_md)
            #plt.savefig(os.path.join(folder_to_save, file_name_md), dpi=300)
            #plt.show()

            title_MSE = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'

            file_name_MSE = 'MSE_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            sns.boxplot(ax=axes[1], x='Partition Size', y='Mean Squared Error', data=df_MSE).set_title(title_MSE)
            #plt.savefig(os.path.join(folder_to_save, file_name_MSE), dpi=300)
            #plt.show()       


            title_SNR = name + ':  ' + 'runs: ' + str(n_runs) + ', ' + str(
                noise) + 'dB noise' + ', ' + str(p) + ' % outliers'
            file_name_SNR = 'SNR_' + ica_method + '_' + name + '_' + str(n_runs) + 'Runs' + '_' + str(
                noise) + 'dB_' + 'p_outliers_' + str(p) +'_Partitionsteps' + '.jpg'
            sns.boxplot(ax=axes[2], x='Partition Size', y='Signal to Noise Ratio\n (Mixed Signals) in dB', data=df_SNR).set_title(title_SNR)
            plt.savefig(os.path.join(folder_to_save, file_name_SNR), dpi=300)
            plt.show()


        if name == "Type scat":
            outlier_type = "impulse"
            mds_ges = np.empty(len(noise) * n_runs)
            theoretical_mds = np.empty(len(noise) * n_runs)
            median_MSEs_ges = np.empty(len(noise) * n_runs)
            median_SNRs_ges = np.empty(len(noise) * n_runs)
            mean_MSEs_ges = np.empty(len(noise) * n_runs)
            mean_SNRs_ges = np.empty(len(noise) * n_runs)
            i=0
            for snr in noise:
                # do a monte-carlo run
                # mds = monte_carlo_run(n_runs, sample_size , ica_method, seed=None, noise_lvl=noise, p_outlier=p, outlier_type=outlier_type, partition= (ps))

                # d = {'Minimum Distance': mds, 'Partition Size': np.repeat(ps, n_runs), '# MC Runs': np.repeat(n_runs, n_runs)}
                # temp = pd.DataFrame(data=d)
                # df = df.append(temp)
                mds, median_MSEs, median_SNRs,mean_MSEs, mean_SNRs, hSNRs = monte_carlo_run(n_runs, sample_size, ica_method, seed=None, noise_lvl=snr,
                                                  p_outlier=p, outlier_type=outlier_type)
                mds_ges[range(i*n_runs,i*n_runs+n_runs)] = mds
                theoretical_mds[range(i*n_runs,i*n_runs+n_runs)] = np.sqrt(1/hSNRs)
                median_MSEs_ges[range(i*n_runs,i*n_runs+n_runs)] = median_MSEs
                median_SNRs_ges[range(i*n_runs,i*n_runs+n_runs)] = median_SNRs
                mean_MSEs_ges[range(i*n_runs,i*n_runs+n_runs)] = mean_MSEs
                mean_SNRs_ges[range(i*n_runs,i*n_runs+n_runs)] = mean_SNRs
                i += 1
            # mds_ges_array = np.asarray(mds_ges).reshape(len(noise) * n_runs)
            # MSEs_ges_array = np.asarray(MSEs_ges).reshape(len(noise) * n_runs)
            # SNRs_ges_array = np.asarray(SNRs_ges).reshape(len(noise) * n_runs)

            d_scatter = {'Median of SNRs (Mixed Signals) in dB': median_SNRs_ges,'Mean of SNRs (Mixed Signals) in dB': mean_SNRs_ges,
                         'Median of MSEs': median_MSEs_ges,'Mean of MSEs': mean_MSEs_ges, 'Minimum Distance': mds_ges, 'Theoretical Minimum Distance (from SNR)': theoretical_mds}
            df_scatter = pd.DataFrame(data=d_scatter)

            sns.set()
            fig, axes = plt.subplots(3, 3)
            fig.canvas.manager.window.showMaximized()
            fig.tight_layout()
            #plt.subplots_adjust(bottom=0.1, hspace=0.5)
            title_scatter = 'Scatterplots of metrics'
            #fig.suptitle(title_scatter)
            file_name_scatter = 'Scatterplot.jpg'

            #sns.boxplot(ax=axes[0], x='Iterations', y='Minimum Distance', data=df_md).set_title(title_md)
            sns.scatterplot(ax = axes[0,0],data=df_scatter, y='Median of MSEs', x='Minimum Distance',)
            sns.scatterplot(ax = axes[0,1],data=df_scatter, y='Median of SNRs (Mixed Signals) in dB', x='Minimum Distance')
            sns.scatterplot(ax = axes[0,2],data=df_scatter, y='Median of SNRs (Mixed Signals) in dB', x='Median of MSEs')
            sns.scatterplot(ax = axes[1,0],data=df_scatter, y='Mean of MSEs', x='Minimum Distance')
            sns.scatterplot(ax = axes[1,1],data=df_scatter, y='Mean of SNRs (Mixed Signals) in dB', x='Minimum Distance')
            sns.scatterplot(ax = axes[1,2],data=df_scatter, y='Mean of SNRs (Mixed Signals) in dB', x='Mean of MSEs')
            sns.scatterplot(ax = axes[2,0],data=df_scatter, x='Theoretical Minimum Distance (from SNR)', y='Minimum Distance')
            sns.scatterplot(ax = axes[2,1],data=df_scatter, y='Theoretical Minimum Distance (from SNR)', x='Minimum Distance')

            plt.savefig(os.path.join(folder_to_save, file_name_scatter), dpi=300)
            plt.show()


    print("--- Success after %s seconds ---" % (time.time() - start_time))
