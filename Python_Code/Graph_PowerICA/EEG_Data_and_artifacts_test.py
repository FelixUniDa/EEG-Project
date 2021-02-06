#%%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from graph_powerICA import*
from PowerICA import*
import datetime
import mne


if __name__ == "__main__":
  ## pow3 seems to be best for eeg data
  nonlin = 'pow3'
  t_start = 6200
  N = 2000
  b = 0.7

  ## total variance of data to be represented by the unmixed components , choose 1,2 or 3-sigma for 
  totvar1 = 95.4
  totvar2 = 95.4

  # load mne sample data of oddball experiment
  sample_data_folder = mne.datasets.sample.data_path()
  sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                              'sample_audvis_filt-0-40_raw.fif')

  # Might try different data sets,from https://mne.tools/stable/overview/datasets_index.html

  raw = mne.io.read_raw_fif(sample_data_raw_file)
  print(raw.info)
  eeg = np.array(raw.get_data(picks=['eeg'])) # pick only eeg channels
  data = eeg[:, t_start:t_start+N]   #return the number of channels, and samplesize as wanted
  data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1  # Normalize EEG data between -1 and 1
  data = data.T


  """
  Event correspondences:

  'Auditory/Left': 1, 
  'Auditory/Right': 2,
  'Visual/Left': 3, 
  'Visual/Right': 4,
  'smiley': 5, 
  'button': 32}
  """


  stim = np.array(raw.get_data(picks=['stim']))[6,t_start:t_start+N]
  stim = stim.T
  print(stim.shape)
  eeg = data.copy()
  fs = 150
  # delorme_type = 'all'
  prop = 0.5
  snr_dB = 3
  artifacts = 5
  type = np.array(['eye', 'muscle', 'linear', 'electric', 'noise'])
  arts = np.zeros((N,5))
  for i in range(0,5):
    data_outl, outlier = add_artifact(data[0::, i], fs, prop=prop, snr_dB=snr_dB,
                                      number=artifacts, type=type[i], seed=None)
    
    # give some more weight to artifacts, otherwise they become negligible
    # place artifacts at 4 channels each, 20 in total (one third of eeg channels)
    data[0::, i] = data[0::, i] + 10*np.real(outlier)
    data[0::, i+10] = data[0::, i+10] + 10*np.real(outlier)
    data[0::, i+40] = data[0::, i+5] + 10*np.real(outlier)
    data[0::, i+25] = data[0::, i+5] + 10*np.real(outlier)
    arts[::,i] = outlier

  white_data1,W_whiten1,n_comp1 = whitening(data.T, type='Mscat', percentile=totvar1/100)
  white_data2,W_whiten2,n_comp2 = whitening(eeg.T, type='sample', percentile=totvar2/100)
  
  corr = np.triu(np.corrcoef(eeg),k=1)
  A1 = np.zeros(corr.shape)
  A1[np.where(corr>0.85)] = corr[np.where(corr>0.85)] #1
  A1 = np.real(A1 + A1.T)
  Ws = np.repeat(A1,n_comp1).reshape(N,N,n_comp1)

  W_graphpower1,_ = Graph_powerICA(np.real(white_data1),nonlin=nonlin,Ws=Ws,b=b)
  W_power1,_ = powerICA(white_data1,'pow3')

  W_graphpower2,_ = Graph_powerICA(np.real(white_data2),nonlin=nonlin,Ws=Ws,b=b)
  W_power2,_ = powerICA(white_data2,'pow3')

  unMixed_graphpower1 = W_graphpower1 @ np.real(white_data1)
  unMixed_power1 = W_power1 @ white_data1

  unMixed_graphpower2 = W_graphpower2 @ np.real(white_data2)
  unMixed_power2 = W_power2 @ white_data2


### plot figures ###
fig1, axs1 = plt.subplots(21, sharex=True)
plt.xlim(xmin=0, xmax=N )
fig2, axs2 = plt.subplots(21, sharex=True)
plt.xlim(xmin=0, xmax=N )
fig3, axs3 = plt.subplots(5, sharex=True)
plt.xlim(xmin=0, xmax=N )
fig4, axs4 = plt.subplots(n_comp1+1, sharex=True)
plt.xlim(xmin=0, xmax=N)
fig5, axs5 = plt.subplots(n_comp1+1, sharex=True)
plt.xlim(xmin=0, xmax=N)
i=0
while(i<20):
    axs1[i].plot(eeg[:, i], lw=1)
    axs1[i].set_ylabel('sig: ' + str(i))
    axs1[20].plot(stim,lw=1)
    axs1[20].set_ylabel('Events')
    fig1.suptitle('Clean EEG Signals - first 20 channels')

    axs2[i].plot(data[:, i], lw=1)
    axs2[i].set_ylabel('sig: ' + str(i))
    axs2[20].plot(stim,lw=1)
    axs2[20].set_ylabel('Events')
    fig2.suptitle('Contaminated EEG Signals - first 20 channels')
    i+=1
i=0
while(i<5):
    axs3[i].plot(arts[:, i], lw=3)
    axs3[i].set_ylabel('sig: ' + str(i))
    fig3.suptitle('Artifacts')
    i+=1
i=0

while(i<n_comp1):
    # axs2[i].plot(mixeddata.T[:, i], lw=1)
    # axs2[i].set_ylabel('sig: ' + str(i))
    # fig2.suptitle('Mixed Signals')


    axs4[i].plot(unMixed_graphpower1.T[:, i], lw=1)
    axs4[i].set_ylabel('sig: ' + str(i))
    axs4[n_comp1].plot(stim,lw=1)
    axs4[n_comp1].set_ylabel('Events')
    fig4.suptitle('Recovered artifacts and eeg signals GraphPowerICA \n( ' + str(totvar1) + '% data represented )')

    axs5[i].plot(unMixed_power1.T[:, i], lw=1)
    axs5[i].set_ylabel('sig: ' + str(i))
    axs5[n_comp1].plot(stim,lw=1)
    axs5[n_comp1].set_ylabel('Events')
    fig5.suptitle('Recovered artifacts and eeg signals PowerICA \n( '+ str(totvar1) +'% data represented )')
    
    i = i+1

fig6, axs6 = plt.subplots(n_comp2+1, sharex=True)
plt.xlim(xmin=0, xmax=N)
fig7, axs7 = plt.subplots(n_comp2+1, sharex=True)
plt.xlim(xmin=0, xmax=N)
i=0    
while(i<n_comp2):
    # axs2[i].plot(mixeddata.T[:, i], lw=1)
    # axs2[i].set_ylabel('sig: ' + str(i))
    # fig2.suptitle('Mixed Signals')


    axs6[i].plot(unMixed_graphpower2.T[:, i], lw=1)
    axs6[i].set_ylabel('sig: ' + str(i))
    axs6[n_comp2].plot(stim,lw=1)
    axs6[n_comp2].set_ylabel('Events')
    fig6.suptitle('Recovered clean eeg signals GraphPowerICA \n( '+ str(totvar2) +'% data represented )')

    axs7[i].plot(unMixed_power2.T[:, i], lw=1)
    axs7[i].set_ylabel('sig: ' + str(i))
    axs7[n_comp2].plot(stim,lw=1)
    axs7[n_comp2].set_ylabel('Events')
    fig7.suptitle('Recovered clean eeg signals PowerICA \n( '+ str(totvar2) +'% data represented )')
    
    i = i+1

date = datetime.datetime.now()
fig4.savefig(fname="EEG-Plots/Recovered_artifacts_and_eeg_signals_GraphPowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
fig5.savefig(fname="EEG-Plots/Recovered_artifacts_and_eeg_signals_PowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
fig6.savefig(fname="EEG-Plots/Recovered_clean_eeg_signals_GraphPowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
fig7.savefig(fname="EEG-Plots/Recovered_clean_eeg_signals_PowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")


plt.show()

