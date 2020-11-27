"""
.. _ex-array-classes:

=====================================
Creating MNE objects from data arrays
=====================================

In this simple example, the creation of MNE objects from
NumPy arrays is demonstrated.
"""
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)
#%%
import numpy as np
import scipy
import sklearn
import scipy.signal as sig
from sklearn.decomposition import PCA, FastICA
from mne.decoding import UnsupervisedSpatialFilter
import matplotlib.pyplot as plt
import mne

###############################################################################
# print(__doc__)
# ## Preprocessing functions
# 
# To get an optimal estimate of the independent components it is advisable to do some pre-processing of the data. In the following the two most important pre-processing techniques are explained in more detail.
# 
def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered =  x - mean 
    return centered, mean

# For the second pre-processing technique we need to calculate the covariance. So lets quickly define it.
def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n

def whiten(X):
     # Calculate the covariance matrix
     coVarM = covariance(X) 
   
     # Single value decoposition
     U, S, V = np.linalg.svd(coVarM)
   
     # Calculate diagonal matrix of eigenvalues
     d = np.diag(1.0 / np.sqrt(S)) 
   
     # Calculate whitening matrix
     whiteM = np.dot(U, np.dot(d, U.T))
  
     # Project onto whitening matrix
     Xw = np.dot(whiteM, X) 
   
     return Xw, whiteM
###############################################################################
# Create arbitrary data
#%%
sfreq = 1000  # Sampling frequency
times = np.arange(0, 10, 0.001)  # Use 10000 samples (10s)
sin = np.sin(np.pi*times*100)  # Multiplied by 10 for shorter cycles
square = sig.square(times*150)
sawtooth = sig.sawtooth(times*100-50,1)
cos = np.cos(np.pi*times*100)

# Numpy array of size 4 X 10000.
data = np.array([sin, square, sawtooth, cos])

# Matrix defining the mixture between the Signals, has a big impact on how well the data can be seperated
mixingmat = np.array([[1,      0.2,    0.1,      0.01],
                      [0.2,      1,    2,      0.1], 
                      [0.1,      2,    1,      0.2],
                      [0.01,    0.1,    0.2,      1]])

mixeddata = mixingmat@data
print("data:","\n",data,"\n","mixeddata","\n",mixeddata)

# Center signals
centered_data, mean_data = center(mixeddata)

# Whiten mixed signals
whitened_data , whiteM = whiten(centered_data)
print(np.cov(whitened_data))


#%%
from PowerICA import *
m, n = whitened_data.shape

# # Initialize random weights
W = scipy.stats.ortho_group.rvs(4)
W = powerICA(whitened_data,'tanh',W)
# W = fastIca(whitened_data)
#fica = sklearn.decomposition.FastICA(n_components=4,whiten=True)
 #Un-mix signals using 
unMixed = whitened_data.T.dot(W)

# # Subtract mean
unMixed = (unMixed.T - mean_data).T

# %%
# Plot input signals (not mixed)
fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(data.T, lw=5)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Source signals', fontsize=25)
ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(mixeddata.T, lw=5)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Mixed signals', fontsize=25)
ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(unMixed, '--', label='Recovered signals', lw=5)
ax.set_xlabel('Sample number', fontsize=20)
ax.set_title('Recovered signals', fontsize=25)
ax.set_xlim(0, 100)

plt.show()





#%%
# Definition of channel types and names.
ch_types = ['mag', 'mag', 'grad', 'grad']
ch_names = ['sin', 'square', 'sawtooth', 'cos']

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

raw.plot(n_channels=4, scalings=scalings, title='Data from arrays',
         show=True, block=True)

# It is also possible to auto-compute scalings
#scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw_mixed.plot(n_channels=4, scalings=scalings, title='Mixed Data from arrays',
         show=True, block=True)

#%%
ica2 = mne.preprocessing.ICA(n_components=4, random_state=0, method='infomax')
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
epochs_data = np.array([[sin,
                        square,
                        sawtooth,
                        cos]])

ch_types = ['mag', 'mag', 'grad', 'grad']
ch_names = ['sin', 'square', 'sawtooth', 'cos']
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


#%%
###############################################################################
# Create epochs by windowing the raw data.

# The events are spaced evenly every 1 second.
duration = 1.

# create a fixed size events array
# start=0 and stop=None by default
events = mne.make_fixed_length_events(raw, event_id, duration=duration)
print(events)

# for fixed size events no start time before and after event
tmin = 0.
tmax = 0.99  # inclusive tmax, 1 second epochs

# create :class:`Epochs <mne.Epochs>` object
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=None, verbose=True)
epochs.plot(scalings='auto', block=True)

###############################################################################
# Create overlapping epochs using :func:`mne.make_fixed_length_events` (50 %
# overlap). This also roughly doubles the amount of events compared to the
# previous event list.

duration = 0.5
events = mne.make_fixed_length_events(raw, event_id, duration=duration)
print(events)
epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, baseline=None,
                    verbose=True)
epochs.plot(scalings='auto', block=True)


# %%



