#%%
import numpy as np
from sklearn.covariance import GraphicalLasso
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from graph_powerICA import*
from PowerICA import*
import datetime
import mne


if __name__ == "__main__":
  # pow3 seems to be best for eeg data, b should be chosen depending on the quality of the graph information / Adjacency
  # for large uncertainties keep it low
  nonlin = 'pow3'
  N_start = 6300
  eeg_fs =150.2
  N = 1000
  b = 0.3

  ## total variance of data to be represented by the unmixed components , choose 1,2 or 3-sigma or any other measure
  totvar1 = 95
  totvar2 = 95
  times = np.linspace(0,N/eeg_fs,N)
  # load mne sample data of oddball experiment
  sample_data_folder = mne.datasets.sample.data_path()
  sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                              'sample_audvis_filt-0-40_raw.fif')

  # Might try different data sets,from https://mne.tools/stable/overview/datasets_index.html

  raw = mne.io.read_raw_fif(sample_data_raw_file)
  print(raw.info)
  eeg = np.array(raw.get_data(picks=['eeg'])) # pick only eeg channels
  data = eeg[:, N_start:N_start+N]   #return the number of channels, and samplesize as wanted
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


  stim = np.array(raw.get_data(picks=['stim']))
  stim = stim[6,N_start:N_start+N]
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

  white_data1,W_whiten1,n_comp1 = whitening(data.T, type='sample', percentile=totvar1/100)
  white_data2,W_whiten2,n_comp2 = whitening(eeg.T, type='sample', percentile=totvar2/100)
  
#   corr = np.triu(np.corrcoef(eeg),k=1)
#   A1 = np.zeros(corr.shape)
#   A1[np.where(corr>0.85)] = 1
#   A1 = np.real(A1 + A1.T)
  # A2 = np.real(A1 + A1.T)
  # G = graphs.Graph(A1)
  # G.set_coordinates(kind='spring')
  # plt.spy(A1)
  # plotting.plot_graph(G, show_edges = True)

  covariance_estimator = GraphicalLasso(max_iter=1000,alpha=0.06,tol=0.001,verbose=1)
  covariance_estimator.fit(data.T)
  connectivity = covariance_estimator.covariance_
  A1 = np.zeros(connectivity.shape)
  A1 = np.triu(connectivity,k=1)
  A1 = A1 + A1.T
  plt.spy(A1)

  Ws1 = np.repeat(A1,n_comp1).reshape(N,N,n_comp1)
  Ws2 = np.repeat(A1,n_comp2).reshape(N,N,n_comp2)

  W_graphpower1,_ = Graph_powerICA(np.real(white_data1),nonlin=nonlin,Ws=Ws1,b=b)
  W_power1,_ = powerICA(white_data1,'pow3')

  W_graphpower2,_ = Graph_powerICA(np.real(white_data2),nonlin=nonlin,Ws=Ws2,b=b)
  W_power2,_ = powerICA(white_data2,'pow3')

  unMixed_graphpower1 = W_graphpower1 @ np.real(white_data1)
  unMixed_power1 = W_power1 @ white_data1

  unMixed_graphpower2 = W_graphpower2 @ np.real(white_data2)
  unMixed_power2 = W_power2 @ white_data2

  sns.set_style("darkgrid")
  fighist,hist = plt.subplots(4,5,figsize=(16,9), sharex=True)
  hist[0,0] = plt.hist(eeg[:, 0])
  hist[0,1] = plt.hist(eeg[:, 1])
  hist[0,2] = plt.hist(eeg[:, 2])
  hist[0,3] = plt.hist(eeg[:, 3])
  hist[0,4] = plt.hist(eeg[:, 4])
  hist[1,0] = plt.hist(data[:, 0])
  hist[1,1] = plt.hist(data[:, 1])
  hist[1,2] = plt.hist(data[:, 2])
  hist[1,3] = plt.hist(data[:, 3])
  hist[1,4] = plt.hist(data[:, 4])
  hist[2,0] = plt.hist(white_data1.T[:, 0])
  hist[2,1] = plt.hist(white_data1.T[:, 1])
  hist[2,2] = plt.hist(white_data1.T[:, 2])
  hist[2,3] = plt.hist(white_data1.T[:, 3])
  hist[2,4] = plt.hist(white_data1.T[:, 4])
  hist[3,0] = plt.hist(unMixed_power2.T[:, 0])
  hist[3,1] = plt.hist(unMixed_power2.T[:, 1])
  hist[3,2] = plt.hist(unMixed_power2.T[:, 2])
  hist[3,3] = plt.hist(unMixed_power2.T[:, 3])
  hist[3,4] = plt.hist(unMixed_power2.T[:, 4])


  ### plot figures ###
  fig1, axs1 = plt.subplots(21,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1] )
  plt.xlabel('t [s]')
  fig2, axs2 = plt.subplots(21,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1] )
  plt.xlabel('t [s]')
  fig3, axs3 = plt.subplots(5,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1] )
  plt.xlabel('t [s]')
  fig4, axs4 = plt.subplots(n_comp1+1,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1])
  plt.xlabel('t [s]')
  fig5, axs5 = plt.subplots(n_comp1+1,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1])
  plt.xlabel('t [s]')
  i=0
  while(i<20):
      axs1[i].plot(times, eeg[:, i], lw=1)
      h = axs1[i].set_ylabel('Ch: ' + str(i),rotation=0)
      axs1[20].plot(times,stim,lw=1)
      axs1[20].set_ylabel('Events',rotation=0)
      fig1.suptitle('Clean EEG Signals - first 20 channels')

      axs2[i].plot(times, data[:, i], lw=1)
      axs2[i].set_ylabel('Ch: ' + str(i),rotation=0)
      axs2[20].plot(times,stim,lw=1)
      axs2[20].set_ylabel('Events',rotation=0)
      fig2.suptitle('Contaminated EEG Signals - first 20 channels')
      i+=1
  i=0
  while(i<5):
      axs3[i].plot(times, arts[:, i], lw=3)
      axs3[i].set_ylabel('Artifact: ' + str(i))
      fig3.suptitle('Artifacts')
      i+=1
  i=0

  while(i<n_comp1):
      # axs2[i].plot(mixeddata.T[:, i], lw=1)
      # axs2[i].set_ylabel('sig: ' + str(i))
      # fig2.suptitle('Mixed Signals')


      axs4[i].plot(times, unMixed_graphpower1.T[:, i], lw=1)
      axs4[i].set_ylabel('IC: ' + str(i),rotation=0)
      axs4[n_comp1].plot(times,stim,lw=1)
      axs4[n_comp1].set_ylabel('Events',rotation=0)
      fig4.suptitle('Recovered artifacts and eeg signals GraphPowerICA \n( ' + str(totvar1) + '% data represented )')

      axs5[i].plot(times, unMixed_power1.T[:, i], lw=1)
      axs5[i].set_ylabel('IC: ' + str(i),rotation=0)
      axs5[n_comp1].plot(times,stim,lw=1)
      axs5[n_comp1].set_ylabel('Events',rotation=0)
      fig5.suptitle('Recovered artifacts and eeg signals PowerICA \n( '+ str(totvar1) +'% data represented )')
      
      i = i+1

  fig6, axs6 = plt.subplots(n_comp2+1,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1])
  plt.xlabel('t [s]')
  fig7, axs7 = plt.subplots(n_comp2+1,figsize=(16,9), sharex=True)
  plt.xlim(xmin=0, xmax=times[-1])
  plt.xlabel('t [s]')
  i=0    
  while(i<n_comp2):
      # axs2[i].plot(mixeddata.T[:, i], lw=1)
      # axs2[i].set_ylabel('sig: ' + str(i))
      # fig2.suptitle('Mixed Signals')


      axs6[i].plot(times, unMixed_graphpower2.T[:, i], lw=1)
      axs6[i].set_ylabel('IC: ' + str(i),rotation=0)
      axs6[n_comp2].plot(times,stim,lw=1)
      axs6[n_comp2].set_ylabel('Events',rotation=0)
      fig6.suptitle('Recovered clean eeg signals GraphPowerICA \n( '+ str(totvar2) +'% data represented )')

      axs7[i].plot(times, unMixed_power2.T[:, i], lw=1)
      axs7[i].set_ylabel('IC: ' + str(i),rotation=0)
      axs7[n_comp2].plot(times,stim,lw=1)
      axs7[n_comp2].set_ylabel('Events',rotation=0)
      fig7.suptitle('Recovered clean eeg signals PowerICA \n( '+ str(totvar2) +'% data represented )')
      
      i = i+1


  date = datetime.datetime.now()
  fig4.savefig(fname="EEG-Plots/Recovered_artifacts_and_eeg_signals_GraphPowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
  fig5.savefig(fname="EEG-Plots/Recovered_artifacts_and_eeg_signals_PowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
  fig6.savefig(fname="EEG-Plots/Recovered_clean_eeg_signals_GraphPowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")
  fig7.savefig(fname="EEG-Plots/Recovered_clean_eeg_signals_PowerICA_( "+ str(totvar2) +"%_data_represented)_b="+str(b)+"_nonlin="+nonlin+"_samples="+str(N)+date.strftime("_%d-%m-%Y_%H-%M-%S")+".png")

  plt.show()


