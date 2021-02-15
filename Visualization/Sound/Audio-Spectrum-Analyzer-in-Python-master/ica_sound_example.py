# sound imports
import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import time

# ica imports
import matplotlib.pyplot as plt
import os
import sys

# system path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'JADE'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils.utils import mixing_matrix, whitening
from Python_Code.JADE.jade import jadeR

# skritp setup
write_to_disc = True
plot_signal = False

# read sound files
fs1, rob1 = read("rob1.wav")
fs2, rob2 = read("rob2.wav")
fs3, vio = read("vio2.wav")
fs4, water = read("water.wav")
# f5, rob3 = read("rob3.wav")
f6, friend = read("water_my_friend.wav")
# friend = friend[:, 0].T
# fill = np.zeros((0, 132000-86730))
# fill[0, 1] = 1
# friend[2] = 3
# friend[0, 86730:132300] = fill

# create data for ICA
data = np.stack([rob1, vio, water, rob2]).T # rob2,friend
c, r = data.shape
MM = mixing_matrix(r, seed=1)
mixdata = MM @ data.T
mixdata_noise = mixdata

# whitening
white_data, W_whiten, W_dewhiten, _ = whitening(mixdata_noise, type='sample')

# perform ICA
W_jade = jadeR(white_data)
W_jade = np.squeeze(np.asarray(W_jade))
unMixed_jade = W_jade @ white_data

# mixed & recovered signals
mixedsignal0 = mixdata.T[:, 0]
mixedsignal1 = mixdata.T[:, 1]
rob1_recovered = unMixed_jade.T[:, 0]
vio_recovered = unMixed_jade.T[:, 1]
water_recovered = unMixed_jade.T[:, 2]
rob2_recovered = unMixed_jade.T[:, 3]
# friend_recovered = unMixed_jade.T[:, 4]


if write_to_disc == True:
    write("mixed_signal0.wav", fs1, mixedsignal0)
    write("mixed_signal1.wav", fs1, mixedsignal1)
    write("rov1_recovered.wav", fs1, rob1_recovered)
    write("vio_recovered.wav", fs1, vio_recovered)
    write("water_recovered.wav", fs1, water_recovered)
    write("rob2_recovered.wav", fs1, rob2_recovered)
    # write("friend_recovered.wav", fs1, friend_recovered)

# play signals
duration_s = 3.0
sd.play(mixedsignal0, fs1)
time.sleep(duration_s)
sd.stop()

sd.play(mixedsignal1, fs1)
time.sleep(duration_s)
sd.stop()

sd.play(rob1_recovered, fs1)
time.sleep(duration_s)
sd.stop()

sd.play(vio_recovered, fs1)
time.sleep(duration_s)
sd.stop()

sd.play(water_recovered, fs1)
time.sleep(duration_s)
sd.stop()

sd.play(rob2_recovered, fs1)
time.sleep(duration_s)
sd.stop()

# sd.play(friend_recovered, fs1)
# time.sleep(duration_s)
# sd.stop()

# plot signals
if plot_signal == True:
    # plot setups
    i = 0
    total_plots = 2
    fig1, axs1 = plt.subplots(r, sharex=True)
    fig2, axs2 = plt.subplots(r, sharex=True)
    fig3, axs3 = plt.subplots(r, sharex=True)
    fig4, axs4 = plt.subplots(r, sharex=True)
    fig5, axs5 = plt.subplots(r, sharex=True)
    fig6, axs6 = plt.subplots(r, sharex=True)
    fig7, axs7 = plt.subplots(r, sharex=True)

    while (i < r):
        # input signals
        axs1[i].plot(data[:, i], lw=3)
        axs1[i].set_ylabel('sig: ' + str(i))
        fig1.suptitle('Input Signals')

        axs2[i].plot(mixdata.T[:, i], lw=3)
        axs2[i].set_ylabel('sig: ' + str(i))
        fig2.suptitle('Mixed Signals')

        axs3[i].plot(mixdata_noise.T[:, i], lw=3)
        axs3[i].set_ylabel('sig: ' + str(i))
        fig3.suptitle('Contaminated Mixed Signals')

        axs4[i].plot(unMixed_power.T[:, i], lw=3)
        axs4[i].set_ylabel('sig: ' + str(i))
        fig4.suptitle('Recovered signals PowerICA')

        axs5[i].plot(unMixed_jade.T[:, i], lw=3)
        axs5[i].set_ylabel('sig: ' + str(i))
        fig5.suptitle('Recovered signals JADE')

        axs6[i].plot(unMixed_radical.T[:, i], lw=3)
        axs6[i].set_ylabel('sig: ' + str(i))
        fig6.suptitle('Recovered signals RADICAL')

        axs7[i].plot(unMixed_coro.T[:, i], lw=3)
        axs7[i].set_ylabel('sig: ' + str(i))
        fig7.suptitle('Recovered signals CoroICA')

        i = i + 1

    plt.show()




