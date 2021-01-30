import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath( __file__ )))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'JADE'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'Compare_ICA_algos'))

import PowerICA
import seaborn as sns
from fast_Radical import *
from jade import jadeR
# from distances import *
import pandas as pd
from coroica import CoroICA
from utils import *




def create_artif(x=10000, fs=1000, eeg_components=1, artifacts=1, type='eye'):



    #eeg_data = create_signal(x=x, c='eeg', ampl=1, eeg_components=eeg_components)
    eeg_data = np.stack([create_signal(f=2, x=x, c='ecg'),
                     create_signal(f=5, x=x, ampl=1, c='cos'),
                     create_signal(f=10, x=x, c='rect'),
                     create_signal(f=25, x=x, c='sawt')]).T

    # parameter for add_artifact function for delorme artefacts
    fs = 1000
    eeg_components = 4
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

    # Mix data before adding artifacts
    c, r = eeg_data.shape
    MM = mixing_matrix(r, None, m=artifacts)
    # print(MM)
    mixdata = (MM @ eeg_data.T).T

    # Add artifacts
    eeg_data_artif = np.zeros_like(mixdata)
    eeg_data_outlier = np.zeros_like(mixdata)
    for i in range(0, eeg_components + artifacts):
        if (i < artifacts):
            data_outl, outlier = add_artifact(mixdata[0::, i], fs, prop=prop, snr_dB=snr_dB,
                                              number=artifacts, type=type[i], seed=None)
            eeg_data_artif[0::, i] = data_outl
            eeg_data_outlier[0::, i] = outlier
        else:
            eeg_data_artif[0::, i] = mixdata[0::, i]
            eeg_data_outlier[0::, i] = eeg_data_outlier[0::, i]

    # plt.plot(eeg_data_artif)
    # plt.show()


    noise_lvl = 1000
    p_outlier = 0.0
    std_outlier = 3
    outlier_type = 'impulse'

    # apply noise and/or create outlier
    mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl),
                                             prop=p_outlier, std=std_outlier, type=outlier_type) for dat in
                              eeg_data_artif])

    noise = (mixdata_noise - eeg_data_artif).T
    eeg_data_outlier = eeg_data_outlier[:, 0:artifacts]
    data_ideal = np.concatenate((eeg_data, eeg_data_outlier), axis=1)
    # centering the data and whitening the data:
    white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise.T, type='sample')
    '''
    if (type == 'all'):
        type = np.array(['eye', 'muscle', 'linear', 'electric', 'noise'])
        artifacts = 5
    else:
        type = np.repeat(type, artifacts)

    if(eeg_components < artifacts):
        artifacts = eeg_components


    eeg_data_artif = np.zeros_like(eeg_data)
    data_artif = np.zeros_like(eeg_data)
    for i in range(0, eeg_components):
        if(i < artifacts):
            data_outl, outlier = add_artifact(eeg_data[0::, i], fs, prop=0.2, snr_dB=3, number=artifacts, type=type[i], seed=1)
            eeg_data_artif[0::, i] = data_outl
            data_artif[0::, i] = outlier
        else:
            eeg_data_artif[0::, i] = eeg_data[0::, i]
            data_artif[0::, i] = outlier

    # plt.plot(eeg_data_artif)
    # plt.show()

    c, r = eeg_data_artif.shape
    MM = mixing_matrix(r, None, m=artifacts)
    # print(MM)
    mixdata = MM @ eeg_data_artif.T

    noise_lvl = 100
    p_outlier = 0.0
    # apply noise and/or create outlier
    mixdata_noise = np.stack(
        [create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl), prop=p_outlier, std=3, type='impulse') for dat
         in mixdata])

    # plt.plot(mixdata_noise.T)
    # plt.show()

    # centering the data and whitening the data:
    white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample', r=1*10**(-5))
    # plt.plot(white_data.T)
    # plt.show()
    '''

    return eeg_data, eeg_data_outlier, mixdata_noise, white_data, data_ideal

if __name__ == "__main__":

    sns.set_theme(style='darkgrid')
    sns.set_context('paper')

    artifacts = 4
    eeg_data, eeg_data_artif, mixdata_noise, artif_white_data, data_ideal = create_artif(x=10000, fs=1000, eeg_components=4, artifacts=artifacts, type='all')
    W_power, _ = PowerICA.powerICA(artif_white_data, 'pow3')
    c = CoroICA(partitionsize=10, groupsize=1000)
    c.fit(artif_white_data.T)
    W = c.V_

    unMixed_power = W_power @ artif_white_data
    #unMixed_power = W @ artif_white_data
    MSE_data = np.mean(MSE(unMixed_power, data_ideal.T))
    print(MSE_data)

    r, c = unMixed_power.shape


    # Plot input signals (not mixed)
    i = 0
    fig1, axs1 = plt.subplots(4)  # , sharex=True)
    fig2, axs2 = plt.subplots(4)  # , sharex=True)
    fig3, axs3 = plt.subplots(r)  # , sharex=True)
    fig4, axs4 = plt.subplots(r)  # , sharex=True)
    # fig5, axs5 = plt.subplots(r)  # , sharex=True)
    # fig6, axs6 = plt.subplots(r)  # , sharex=True)

    while (i < r-artifacts):
        # input signals
        axs1[i].plot(eeg_data[:, i], lw=1)
        axs1[i].set_ylabel('sig: ' + str(i))
        fig1.suptitle('Input Signals')

        axs2[i].plot(eeg_data_artif[:, i], lw=1)
        axs2[i].set_ylabel('sig: ' + str(i))
        fig2.suptitle('artifact signals')
        i = i + 1

    i = 0
    while (i < r):
        axs3[i].plot(mixdata_noise[:, i], lw=1)
        axs3[i].set_ylabel('sig: ' + str(i))
        fig3.suptitle('Mixed signals')

        axs4[i].plot(unMixed_power.T[:, i], lw=1)
        axs4[i].set_ylabel('sig: ' + str(i))
        fig4.suptitle('Recovered signals PowerICA')

        i = i + 1

    plt.show()

