#%%
import numpy as np
import scipy
import scipy.signal as sig
from sklearn.decomposition import FastICA
#from mne.decoding import UnsupervisedSpatialFilter
import matplotlib.pyplot as plt
#import mne
from utils import *

###############################################################################
# print(__doc__)
# ## Preprocessing functions
# 
# To get an optimal estimate of the independent components it is advisable to do some pre-processing of the data. In the following the two most important pre-processing techniques are explained in more detail.
# 
###############################################################################
# Create arbitrary data
#%%
sfreq = 1000  # Sampling frequency
times = np.arange(0, 1000, 0.05)  # Use 10000 samples (10s)
sig1 = np.sin(np.pi*times)
sig2 = np.sin(np.pi*times*0.5)
#sig2 = sig.sawtooth(times*2-50,1)
#sig3 = sig.square(times*2,1)
sig3 = np.cos(np.pi*times*4) 
sig4 = np.cos(np.pi*times*2) 

# Numpy array of size 4(2) X 10000.
data = np.array([sig1,sig2,sig3,sig4])
n_comp,n_samples =np.shape(data)
# Matrix defining the mixture between the Signals, has a big impact on how well the data can be seperated
mixingmat = np.random.rand(n_comp,n_comp)
#np.random.rand(n_comp,n_comp)

mixeddata = mixingmat@data

# Center signals
centered_data, mean_data = center(mixeddata)

# Whiten mixed signals
whitened_data = whiten(centered_data)
print(np.cov(whitened_data))


#%%
from PowerICA import *
m, n = whitened_data.shape

# # Initialize random weights
W, _ = powerICA(whitened_data,'tanh', seed=1)
print(W)
# W = fastIca(whitened_data)
#fica = sklearn.decomposition.FastICA(n_components=4,whiten=True)
 #Un-mix signals using 
unMixed = W @ whitened_data
print(unMixed, np.shape(unMixed))
# # add mean
# unMixed = (unMixed.T + mean_data).T

# Plot input signals (not mixed)
fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(data.T, lw=3)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Source signals', fontsize=25)
ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(mixeddata.T, lw=3)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Mixed signals', fontsize=25)
ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(unMixed.T, label='Recovered signals', lw=3)
ax.set_xlabel('Sample number', fontsize=20)
ax.set_title('Recovered signals', fontsize=25)
ax.set_xlim(0, 100)

plt.show()


'''
##############################################################################
#The following Part deals with the implementation of artificial signals
#into the mne framework by creating a "raw"-instance and how unsupervised 
#spatial filters can be used to work with the toolbox
##############################################################################

#%%
# Definition of channel types and names.
ch_types = ['mag','grad']
ch_names = ['sig1', 'sig2']

###############################################################################
# Create an :class:`info <mne.Info>` object.

# It is also possible to use info from another raw object.
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#%%
###############################################################################
# Create a dummy :class:`mne.io.RawArray` object
raw = mne.io.RawArray(data, info)
raw_mixed = mne.io.RawArray(mixeddata, info)

# Scaling of the figure.
# For actual EEG/MEG data different scaling factors should be used.
scalings = {'mag': 2, 'grad': 2}

#raw.plot(n_channels=4, scalings=scalings, title='Data from arrays',
#         show=True, block=True)

# It is also possible to auto-compute scalings
#scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw_mixed.plot(n_channels=n_comp, scalings=scalings, title='Mixed Data from arrays',
         show=True, block=True)

#%%
ica2 = mne.preprocessing.ICA(n_components=n_comp, random_state=0, method='fastica')
ica2.fit(raw_mixed)
ica2.plot_sources(raw_mixed)


#%%
###############################################################################
# EpochsArray
event_id = 1  # This is used to identify the events.
# First column is for the sample number.
events = np.array([[1, 0, event_id]])  # List of three arbitrary events

# Here a data set of 700 ms epochs from 2 channels is
# created from sin and cos data.
# Any data in shape (n_epochs, n_channels, n_times) can be used.
epochs_data = np.array([[sig1,
                        sig2]])

ch_types = ['mag', 'grad']
ch_names = ['sig1', 'sig2']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

epochs = mne.EpochsArray(epochs_data, info=info, events=events,
                         event_id={'arbitrary': 1})

picks = mne.pick_types(info, meg=True, eeg=False, misc=False)

epochs.plot(picks=picks, scalings='auto', show=True, block=True)

#%%
###############################################################################
# principal component analysis using sklearn.decomposition package
# %%
# independent component analysis using sklearn.decomposition package
ica = UnsupervisedSpatialFilter(FastICA(4), average=False)
ica_data = ica.fit_transform(epochs.get_data())
ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      info, tmin=0)
ev1.plot(show=False, window_title='ICA', time_unit='s')
'''