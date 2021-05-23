#%%
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'JADE'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'Python_Code', 'Compare_ICA_algos'))

from sklearn.covariance import GraphicalLasso
import numpy as np
import matplotlib.pyplot as plt

from graph_powerICA import*
# import depending on machine
# from Python_Code.PowerICA_Python.PowerICA import*
# from utils.utils import *
from utils import *
from PowerICA import*
from scipy import stats




def genrAs(n, mc=0.01):
  Il = np.ones((n, n))
  Il = np.triu(Il, k=1)
  
  lt = np.where(Il.flatten()>0.5)
  
  ind = np.random.binomial(1, mc, int(np.ceil(n*(n-1)/2)))
  
  A = np.zeros((n*n)) # weight/shift matrix
  lt = lt[0][np.where(ind!=0)]
  A[lt] = 1
  A = A.reshape(n,n)
  A = A + A.T

  return A


def GenWs(A, e1, e2, w):
  n = np.shape(A)[0]
  A = np.triu(A, k=1)
  Il = np.ones((n, n))
  Il = np.triu(Il, k=1)
  
  nz = np.where(np.abs(A.flatten())>0.0001)[0]
  l = len(nz)
  lt = np.where(Il.flatten()>0.5)[0]
  z = np.setdiff1d(lt, nz)
  
  W = A
  W[np.where(np.abs(W) > 0.0001)] = w
  W[np.where(np.abs(W) <= 0.0001)] = 0
  W = W.flatten()
  ind1 = np.random.binomial(1, e1, len(nz))
  ind2 = np.random.binomial(1, e2, len(z))
  
  W[nz[np.where(ind1!=0)]] = 0  
  
  W[z[np.where(ind2!=0)]] = w
  
  W = W.reshape(n, n)
  W = W+W.T
  return W


#%%
if __name__ == "__main__":
  nonlins = ['gaus']
  #nonlins = ['gaus']
  bs = [0.7]  
  Signaltypes = ["GMA"]
  for b in bs:
    runs = 1
    mds_power = np.zeros(runs)
    mds_graphpower = np.zeros(runs)
    for nonlin in nonlins:
      for Signaltype in Signaltypes:
        for i in range(0,runs):
          print("run:", i)
          N = 500
          P = 4
          

          if Signaltype == "GMA" :
            ### Set up experiment as in R Code ###
            ### Erdos Renyi Graph ###
            np.random.seed(1996)
            A1 = genrAs(N)
            A2 = A1@A1
            Ws = np.array([A1,A1,A1,A1,A2,A2,A2,A2]).reshape(N,N,2*P)
            ### Graph Gaussian Moving Average Signals ###
            e = np.zeros((N,P))
            e[:,0] = np.random.standard_t(5,N)
            e[:,1] = np.random.standard_t(10,N)
            e[:,2] = np.random.standard_t(15,N)
            e[:,3] = np.random.randn(N)
            print(np.cov(e.T))
            plt.figure()
            plt.spy(A1)


            theta = np.array([0.02,0.04,0.06,0.08])
            data = np.zeros((N, P))
            for j in range(0,P):
              data[:, j] = e[:, j]+theta[j]*Ws[:, :, j]@e[:, j]
            

          elif Signaltype == "Standard":
            data = np.stack([create_signal(f=2,c='ecg',x=N),
                        create_signal(f=5,c='cos',x=N),
                        create_signal(f=10,c='rect',x=N), 
                        create_signal(f=25,c='sawt',x=N)]).T
          
            # corr = np.triu(np.corrcoef(data),k=1)
            # A1 = np.zeros(corr.shape)
            # A1[np.where(corr>0.7)] = 1
            # A1 = A1 #+ A1.T
            # A2 = A1 + A1.T
            # #plt.spy(A1)
            # Ws = np.array([A1,A1,A1,A1]).reshape(N,N,4)
            # Ws2 = np.array([A2,A2,A2,A2]).reshape(N,N,4)
#%%
          ### Mix Signals and whiten ###
          mixmat = mixing_matrix(P)
          mixeddata = mixmat @ data.T
#%%
          part_corr_temp = partial_corr(mixeddata)
#%%
          part_corr = np.copy(part_corr_temp) - np.eye(N)
          part_corr = 0.5*np.log((1.00001+part_corr)/(1.00001-part_corr))
          print(np.count_nonzero(part_corr_temp==-1.0001))
#%%
          
          for ii in range(N):
            for jj in range(N):
              if np.abs(part_corr[ii, jj]) > stats.norm.ppf(0.8, loc=0, scale=1):
                part_corr[ii, jj] = 1
              else:
                part_corr[ii, jj] = 0
#%%   
          #mixeddata = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=20),prop=0.00,type='impulse',std=100) for dat in mixeddata])
          white_data, W_whiten, n_comp = whitening(mixeddata, type='sample', percentile=0.999999)
          #print(part_corr)
#%%
          # covariance_estimator = GraphicalLasso(max_iter=1000,alpha=1.5,tol=0.001,verbose=1)
          # covariance_estimator.fit(mixeddata)
          # connectivity = covariance_estimator.covariance_
          
          B1 = part_corr
          #A1 = A1 + A1.T
          plt.figure()
          plt.spy(B1)
          Ws1 = np.repeat(B1, n_comp).reshape(N, N, n_comp)
          fp = np.count_nonzero(B1[np.where(A1==0)])
          tp = np.sum(B1[np.where(np.abs(A1) > 0)])
          fn = np.sum(np.abs(B1[np.where(np.abs(A1) > 0)]-1))
          tn = np.sum(np.abs(B1[np.where(A1 == 0)]-1))

          false_discovery_rate = fp/(fp + tp)
          sensitivity = tp/(tp+fn)
          specificity = tn/(tn+fp) 
          print(false_discovery_rate,sensitivity,specificity)
#%%
          W_graphpower, _ = Graph_powerICA(white_data, nonlin='gaus', Ws=Ws, b=0.7)
          W_power, _ = powerICA(white_data,'pow3')

          unMixed_graphpower = W_graphpower @ white_data
          unMixed_power = W_power @ white_data

          mds_power[0] = md(W_whiten@mixmat,W_power)
          mds_graphpower[0] = md(W_whiten@mixmat,W_graphpower)
          mse_graphpower = MSE(unMixed_graphpower,data.T)
          print(mds_power, mds_graphpower, mse_graphpower)
      #%%
        # print(np.mean(mds_graphpower),np.mean(mds_power))
        # fig1, axs1 = plt.subplots(1,2, sharey=True)  
        # axs1[0].boxplot(mds_power)
        # axs1[0].set_xlabel("PowerICA")
        # axs1[1].boxplot(mds_graphpower)
        # axs1[1].set_xlabel("GraphPowerICA")
        # #fig1.savefig(fname="b="+str(b)+"_nonlin="+nonlin+"_Signaltype="+Signaltype+".png")
        # plt.show()

#%%
        ### plot figures ###
        fig1, axs1 = plt.subplots(P, sharex=True)
        fig2, axs2 = plt.subplots(P, sharex=True)
        fig3, axs3 = plt.subplots(P, sharex=True)
        fig4, axs4 = plt.subplots(P, sharex=True)
        fig5, axs5 = plt.subplots(P, sharex=True)
        k = 0

        while(k<P):
            # input signals
          axs1[k].plot(data[:, k], lw=3)
          axs1[k].set_ylabel('sig: ' + str(k))
          fig1.suptitle('Input Signals')

          axs2[k].plot(mixeddata.T[:, k], lw=3)
          axs2[k].set_ylabel('sig: ' + str(k))
          fig2.suptitle('Mixed Signals')

          # axs3[i].plot(mixdata_noise.T[:, i], lw=3)
          # axs3[i].set_ylabel('sig: ' + str(i))
          # fig3.suptitle('Contaminated Mixed Signals')

          axs4[k].plot((unMixed_graphpower).T[:, k], lw=3)
          axs4[k].set_ylabel('sig: ' + str(k))
          fig4.suptitle('Recovered signals GraphPowerICA')

          axs5[k].plot((unMixed_power).T[:, k], lw=3)
          axs5[k].set_ylabel('sig: ' + str(k))
          fig5.suptitle('Recovered signals PowerICA')
          
          k = k+1
        plt.show()
# %%

