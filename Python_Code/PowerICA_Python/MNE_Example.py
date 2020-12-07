#%%
import numpy as np
import scipy.signal as sig
import mne
from PowerICA import *
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
times = np.arange(0, 100, 0.05)
sig1 = np.sin(np.pi*times)+0.5*np.random.rand(2000)
sig2 = 0.5*np.random.randn(2000)
sig3 = sig.square(times*2)+0.3*np.random.standard_gamma(1,2000)
sig4 = np.cos(np.pi*times*2)+0.5*np.random.rand(2000)

data = np.array([sig1,sig2,sig3,sig4])
n_comp,n_samples =np.shape(data)
# Matrix defining the mixture between the Signals, has a big impact on how well the data can be seperated
mixingmat = np.random.rand(n_comp,n_comp)
#np.random.rand(n_comp,n_comp)

mixeddata = mixingmat@data
print("data:","\n",data,"\n","mixeddata","\n",np.shape(mixeddata))

#%%
# Definition of channel types and names.
ch_types = ['mag','mag','grad','grad']
ch_source = ['sig1', 'sig2','sig3','sig4']
ch_mixed = ['Mix1', 'Mix2','Mix3','Mix4']
ch_ica = ['IC1', 'IC2','IC3','IC4']
###############################################################################
# Create an :class:`info <mne.Info>` object.

# It is also possible to use info from another raw object.
info_source = mne.create_info(ch_names=ch_source, sfreq=sfreq, ch_types=ch_types)
info_mixed = mne.create_info(ch_names=ch_mixed, sfreq=sfreq, ch_types=ch_types)
info_ica = mne.create_info(ch_names=ch_ica, sfreq=sfreq, ch_types=ch_types)
#%%
###############################################################################
# Create a dummy :class:`mne.io.RawArray` object
raw_source = mne.io.RawArray(data, info_source)
raw_mixed = mne.io.RawArray(mixeddata, info_mixed)

mixed_array = raw_mixed.get_data()
whitened_mixed = whiten(mixed_array)
W,flg = powerICA(whitened_mixed,'tanh',None)
unmixed_array = W @ whitened_mixed

raw_ica = mne.io.RawArray(unmixed_array,info_ica)
# Scaling of the figure.
# For actual EEG/MEG data different scaling factors should be used.
scalings = {'mag': 2, 'grad': 2}

#raw.plot(n_channels=4, scalings=scalings, title='Data from arrays',
#         show=True, block=True)

# It is also possible to auto-compute scalings
#scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw_source.plot(n_channels=n_comp, scalings=scalings, title='Source Signals',
         show=True, block=True)
raw_mixed.plot(n_channels=n_comp, scalings=scalings, title='Mixed Signals',
         show=True, block=True)
raw_ica.plot(n_channels=n_comp, scalings=scalings, title='Independent Components',
          show=True, block=True)

