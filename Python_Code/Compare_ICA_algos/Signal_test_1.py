#%%
import numpy as np
import scipy
import scipy.signal as sig
from sklearn.decomposition import fastica
from mne.decoding import UnsupervisedSpatialFilter
import matplotlib.pyplot as plt
import mne
from utils import *
import PowerICA
import jade

# create example signals:
data = np.stack([create_signal(ampl=4, c='ecg'),
                 create_signal(ampl=4,c='cos'),
                 create_signal(c='rect'),
                 create_signal(c='sawt')]).T

data_test = create_signal(ampl=1, c='all')


# Standardize data
#data /= data.std(axis=0)  # Standardize data

# create mixing matrix and mixed signals
c, r = data.shape
MM = mixing_matrix(r)
mixdata = MM@data.T

#apply noise
mixdata_noise = np.stack([apply_noise(dat, SNR_dB=10) for dat in mixdata])

# centering the data and whitening the data:
X_data = whitening(mixdata_noise, type='sample')

# perform PowerICA
W, _ = PowerICA.powerICA(X_data, 'tanh')

# perform Jade
W_jade = jade.jadeR(mixdata_noise)
W_jade = np.squeeze(np.asarray(W_jade))

#Perform fastICA
K, W_fast, S = fastica(mixdata_noise)
#W_fast = sklearn.decomposition.FastICA(n_components=4,whiten=True)

# Un-mix signals using
unMixed = W @ X_data
unMixed_jade = W_jade @ X_data
unMixed_fast= W_fast @ X_data

# Plot input signals (not mixed)
i = 0
fig1, axs1 = plt.subplots(r, sharex=True)
fig2, axs2 = plt.subplots(r, sharex=True)
fig3, axs3 = plt.subplots(r, sharex=True)
fig4, axs4 = plt.subplots(r, sharex=True)
fig5, axs5 = plt.subplots(r, sharex=True)

while(i<r):
    # input signals
    axs1[i].plot(data[:, i], lw=3)
    axs1[i].set_ylabel('sig: ' + str(i))
    fig1.suptitle('Input Signals')


    axs2[i].plot(mixdata.T[:, i], lw=3)
    axs2[i].set_ylabel('sig: ' + str(i))
    fig2.suptitle('Mixed signals')


    axs3[i].plot(unMixed.T[:, i], lw=3)
    axs3[i].set_ylabel('sig: ' + str(i))
    fig3.suptitle('Recovered signals PowerICA')

    axs4[i].plot(unMixed_jade.T[:, i], lw=3)
    axs4[i].set_ylabel('sig: ' + str(i))
    fig4.suptitle('Recovered signals JADE')

    axs5[i].plot(unMixed_fast.T[:, i], lw=3)
    axs5[i].set_ylabel('sig: ' + str(i))
    fig5.suptitle('Recovered signals FastICA')

    i = i+1

plt.show()

'''
ax1 = fig1.add_subplot(1, i, figsize=[18, 5])
ax.plot(data, lw=3)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Source signals', fontsize=25)
#ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(mixdata.T, lw=3)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Mixed signals', fontsize=25)
#ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(unMixed.T, label='Recovered signals', lw=3)
ax.set_xlabel('Sample number', fontsize=20)
ax.set_title('Recovered signals', fontsize=25)
#ax.set_xlim(0, 100)

plt.show()
'''


