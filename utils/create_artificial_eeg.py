from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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




def create_artif(x=10000, fs=1000, eeg_components=1, artifacts=1, type='eye'):

    eeg_data = create_signal(x=x, c='eeg', ampl=1, eeg_components=eeg_components)
    if (type == 'all'):
        type = np.array(['eye', 'muscle', 'linear', 'electric', 'noise'])
        artifacts = 5
    else:
        type = np.repeat(type, artifacts)

    if(eeg_components < artifacts):
        artifacts = eeg_components


    eeg_data_artif = np.zeros_like(eeg_data)
    for i in range(0, eeg_components):
        if(i < artifacts):
            eeg_data_artif[0::, i] = add_artifact(eeg_data[0::, i], fs,  prop=0.1, snr_dB=3, number=artifacts, type=type[i], seed=None)
        else:
            eeg_data_artif[0::, i] = eeg_data[0::, i]

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

    # centering the data and whitening the data:
    white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample')

    return eeg_data, eeg_data_artif, mixdata_noise, white_data


if __name__ == "__main__":

    artifacts = 3
    eeg_data, eeg_data_artif, mixdata_noise, artif_white_data = create_artif(x=10000, fs=1000, eeg_components=3, artifacts=artifacts, type='all')
    W_power, _ = PowerICA.powerICA(artif_white_data, 'pow3')
    unMixed_power = W_power @ artif_white_data

    r, c = unMixed_power.shape


    # Plot input signals (not mixed)
    i = 0
    fig1, axs1 = plt.subplots(r)  # , sharex=True)
    fig2, axs2 = plt.subplots(r)  # , sharex=True)
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
        axs3[i].plot(mixdata_noise.T[:, i], lw=1)
        axs3[i].set_ylabel('sig: ' + str(i))
        fig3.suptitle('Mixed signals')

        axs4[i].plot(unMixed_power.T[:, i], lw=1)
        axs4[i].set_ylabel('sig: ' + str(i))
        fig4.suptitle('Recovered signals PowerICA')

        i = i + 1

    plt.show()
