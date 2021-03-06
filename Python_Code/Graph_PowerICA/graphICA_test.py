#%%
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from graph_powerICA import*
import pygsp
import seaborn as sns
import pandas as pd

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'JADE'))
sys.path.append(os.path.join(BASE_DIR, 'utils', 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'Compare_ICA_algos'))

from utils.utils import *
from PowerICA import *



def genrAs(n, mc=0.01):
  Il = np.ones((n, n))
  Il = np.triu(Il,k=1)
  
  lt = np.where(Il.flatten()>0.5)
  
  ind = np.random.binomial(1,mc,int(np.ceil(n*(n-1)/2)))
  
  A = np.zeros((n*n)) # weight/shift matrix
  lt = lt[0][np.where(ind!=0)]
  A[lt] = 1
  A = A.reshape(n,n)
  A = A + A.T

  return A


def GenWs(A, e1, e2, w):
  n = np.shape(A)[0]
  A = np.triu(A,k=1 )
  Il = np.ones((n, n))
  Il = np.triu(Il,k=1 )
  
  nz = np.where(np.abs(A.flatten())>0.0001)[0]
  l = len(nz)
  lt = np.where(Il.flatten()>0.5)[0]
  z = np.setdiff1d(lt,nz)
  
  W = A
  W[np.where(np.abs(W)>0.0001)] = w
  W[np.where(np.abs(W)<=0.0001)] = 0
  W = W.flatten()
  ind1 = np.random.binomial(1,e1,len(nz))
  ind2 = np.random.binomial(1,e2,len(z))
  
  W[nz[np.where(ind1!=0)]] = 0  
  
  W[z[np.where(ind2!=0)]] = w
  
  W = W.reshape(n,n)
  W = W+W.T
  return W



if __name__ == "__main__":
  runs = 25
  mds_power = np.zeros(runs)
  mse_power = np.zeros(runs)
  mse_Graphpower = np.zeros(runs)
  mds_graphpower = np.zeros(runs)
  bins = [0.7]
  N = 1000

  for b in bins:
    for i in range(0, runs):
      print("run:", i)
      P = 1
      ### Set up experiment as in R Code ###
      ### Erdos Renyi Graph ###
      #A1 = genrAs(N)
      #A2 = A1@A1
      #Ws1 = np.array([A1,A1,A1,A1,A2,A2,A2,A2]).reshape(1000,1000,8)

      #W = GenWs(A1,0.8,0.2,1) #parameters taken from RCode, even the paperversion wird Hops genommen...

      # G1 = graphs.ErdosRenyi(N,p=0.01).W.toarray() ### works aswell but takes forever to build graph
      # G2 = G1 @ G1
      # Ws1 = np.array([G1,G1,G1,G1,G2,G2,G2,G2]).reshape(1000,1000,8)

      # A3 = genrAs(N)
      # A4 = A3@A3
      #Ws2 = np.array([A3,A3,A3,A3,A4,A4,A4,A4]).reshape(1000,1000,8)


      data = np.stack([create_signal(f=2,c='ecg',x=N),
                  create_signal(f=5,c='cos',x=N),
                  create_signal(f=10,c='rect',x=N),
                  create_signal(f=25,c='sawt',x=N),
                  # create_signal(f=13, c='ecg', x=1000),
                  # create_signal(f=17, c='cos', x=1000),
                  # create_signal(f=29, c='rect', x=1000),
                  # create_signal(f=33, c='sawt', x=1000),
                       ]).T

      # mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=noise_lvl),
      #                                          std=std_outlier, prop=p_outlier, type=outlier_type) for dat in mixdata])


      # ### Graph Gaussian Moving Average Signals ###
      # e = np.zeros((N,P))
      # e[:,0] = np.random.standard_t(5,N)
      # e[:,1] = np.random.standard_t(10,N)
      # e[:,2] = np.random.standard_t(15,N)
      # e[:,3] = np.random.randn(N)
      #
      # theta = np.array([0.02,0.04,0.06,0.08])
      # S = np.zeros((N, P))
      # for j in range(0,P):
      #   S[:,j] = e[:,j]+theta[j]*Ws1[:,:,j]@e[:,j]
      #
      ### Mix Signals and whiten ###
      mixmat = mixing_matrix(4, seed=None)
      mixeddata = mixmat @ data.T
      mixdata_noise = np.stack([create_outlier(apply_noise(dat, type='white', SNR_dB=100),
                                               std=100, prop=0.00, type='impulse') for dat in mixeddata])


      #### running weighted median smoother ####
      r, c = mixdata_noise.shape
      win_len_half = 5
      mixdata_noise_ms = np.zeros_like(mixdata_noise)
      for rr in range(0, r):
        for cc in range(win_len_half, int(c-win_len_half)):
          x_win = mixdata_noise[rr, (cc-win_len_half):(cc+win_len_half)]
          mixdata_noise_ms[rr, cc] = np.median(np.hamming(2*win_len_half)*x_win)


      white_data, W_whiten, W_dewhiten,_ = whitening(mixdata_noise, type='sample')

      # plt.plot(white_data.T)
      # plt.show()
      C = np.triu(np.corrcoef(data), k=4) #Mscat(mixeddata, 'Huber')
      thres = 0.7
      C[np.where(abs(C < thres))] = 0
      C[np.where(abs(C >= thres))] = 1
      C = C + C.T
      edges = np.sum(C[np.where(C == 1)])
      print(edges)

      Ws1 = np.stack([C, C, C, C]).reshape(N, N, 4)

      W_graphpower,_ = Graph_powerICA(white_data, 'gaus', Ws1, b=b, radical=0)
      W_power,_ = powerICA(white_data, 'pow3')

      unMixed_graphpower = W_graphpower @ white_data
      unMixed_power = W_power @ white_data

      mds_power[i] = md(W_whiten@mixmat, W_power)
      mds_graphpower[i] = md(W_whiten@mixmat,W_graphpower)
      mse_power[i] = np.mean(MSE(unMixed_power, data.T))
      mse_Graphpower[i] = np.mean(MSE(unMixed_graphpower, data.T))


    d_md_power = {'Minimum Distance': mds_power, 'Sample Size': N}
    d_md_graphPower = {'Minimum Distance': mds_graphpower, 'Sample Size': N}
    d_mse_power = {'Mean Squared Error': mse_power, 'Sample Size': N}
    d_mse_graphPower = {'Mean Squared Error': mse_Graphpower, 'Sample Size': N}

    df_md_power = pd.DataFrame(data=d_md_power)
    df_md_graphPower = pd.DataFrame(data=d_md_graphPower)
    df_mean_MSE_power = pd.DataFrame(data=d_mse_power)
    df_mean_MSE_graphPower = pd.DataFrame(data=d_mse_graphPower)


    print(np.mean(mds_power), np.mean(mds_graphpower))
    print(np.mean(mse_power), np.mean(mse_Graphpower))

    sns.set(style='darkgrid', )
    sns.set(rc={'axes.labelsize': 20})
    #plt.clf()
    #sns.set_context('paper')
    fig, axes = plt.subplots(2, 2, figsize=(12, 16), sharey=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.5)
    title = 'samples: ' + str(N) #+ 'Lambda: ' + str(b) +
    fig.suptitle(title, fontsize=20)

    title_Power = 'Power ICA'
    title_GraphPower = 'Graph-Power ICA'

    #plt.ylim(0, 0.7)
    sn1 = sns.boxplot(ax=axes[0, 0], x='Sample Size', y='Minimum Distance', data=df_md_power).set_title(title_Power, fontsize=20)
    sn2 = sns.boxplot(ax=axes[0, 1], x='Sample Size', y='Minimum Distance', data=df_md_graphPower).set_title(title_GraphPower, fontsize=20)

    #plt.ylim(0, 0.1)
    sn3 = sns.boxplot(ax=axes[1, 0], x='Sample Size', y='Mean Squared Error', data=df_mean_MSE_power)#.set_title(title_Power)
    sn4 = sns.boxplot(ax=axes[1, 1], x='Sample Size', y='Mean Squared Error', data=df_mean_MSE_graphPower)#.set_title(title_GraphPower)


    # fig1, axs1 = plt.subplots(1, 2, sharey=True)
    # axs1[0].boxplot(mds_power)
    # axs1[0].set_xlabel("powerICA")
    # axs1[0].set_ylabel("Minimum Distance")
    # axs1[1].boxplot(mds_graphpower)
    # axs1[1].set_xlabel("Graph-powerICA")
    # fig1.suptitle('Lambda: ' + str(b) + ', samples: ' + str(N))

    plt.show()
#%%

  i = 0
  fig1, axs1 = plt.subplots(4)
  fig2, axs2 = plt.subplots(4)
  fig3, axs3 = plt.subplots(4)

  while (i < 4):
    # input signals
    axs1[i].plot(data.T[i, :], lw=1)
    axs1[i].set_ylabel('sig: ' + str(i))
    fig1.suptitle('Input Signals')


    axs2[i].plot(unMixed_power[i, :], lw=1)
    axs2[i].set_ylabel('sig: ' + str(i))
    fig2.suptitle('Unmixed Signals PowerICA')
    #axs2[i].set_ylim(ymin=-2, ymax=2)
    axs2[i].set_xlabel("samples")

    axs3[i].plot(unMixed_graphpower[i, :], lw=1)
    axs3[i].set_ylabel('sig: ' + str(i))
    fig3.suptitle('Unmixed Signals Graph PowerICA')
    #axs3[i].set_ylim(ymin=-2, ymax=2)
    axs3[i].set_xlabel("samples")

    i = i + 1

  plt.show()

