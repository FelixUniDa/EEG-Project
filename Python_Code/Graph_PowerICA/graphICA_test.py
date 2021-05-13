#%%
from sklearn.covariance import GraphicalLasso
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from graph_powerICA import*
from PowerICA import*


def genrAs(n, mc=0.01):
  Il = np.ones((n, n))
  Il = np.triu(Il,k=1 )
  
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
  nonlins = ['gaus']
  #nonlins = ['gaus']
  bs = [0.7]  
  Signaltypes= ["Standard"]
  for b in bs:
    runs = 1
    mds_power = np.zeros(runs)
    mds_graphpower = np.zeros(runs)
    for nonlin in nonlins:
      for Signaltype in Signaltypes:
        for i in range(0,runs):
          print("run:",i)
          N = 1000
          P = 4
          
          #W = GenWs(A1,0.8,0.2,1) #parameters taken from RCode, even the paperversion wird Hops genommen...
        
          # G1 = graphs.ErdosRenyi(N,p=0.01).W.toarray() ### works aswell but takes forever to build graph
          # G2 = G1 @ G1
          # Ws1 = np.array([G1,G1,G1,G1,G2,G2,G2,G2]).reshape(1000,1000,8)
          
          # A3 = genrAs(N)
          # A4 = A3@A3
          #Ws2 = np.array([A3,A3,A3,A3,A4,A4,A4,A4]).reshape(1000,1000,8)
        

        # data = np.stack([create_signal(f=2,c='ecg',x=1000),
        #               create_signal(f=5,c='cos',x=1000),
        #               create_signal(f=10,c='rect',x=1000), 
        #               create_signal(f=25,c='sawt',x=1000)]).T

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
           


            theta = np.array([0.02,0.04,0.06,0.08])
            data = np.zeros((N, P))
            for j in range(0,P):
              data[:,j] = e[:,j]+theta[j]*Ws[:,:,j]@e[:,j]   
            

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

          ### Mix Signals and whiten ###
          mixmat = mixing_matrix(P, m = 15)
          mixeddata = mixmat @ data.T
          #mixeddata = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=20),prop=0.00,type='impulse',std=100) for dat in mixeddata])
          white_data,W_whiten,n_comp= whitening(mixeddata, type='sample', percentile = 0.99)
          print(n_comp)
          covariance_estimator = GraphicalLasso(max_iter=1000,alpha=1.5,tol=0.01,verbose=1)
          covariance_estimator.fit(mixeddata)
          connectivity = covariance_estimator.covariance_
          A1 = np.zeros(connectivity.shape)
          A1 = corr = np.triu(connectivity,k=1)
          A1 = A1 + A1.T

          Ws1 = np.repeat(A1,n_comp).reshape(N,N,n_comp)

          W_graphpower,_ = Graph_powerICA(white_data,nonlin=nonlin,Ws=Ws1,b=b)
          W_power,_ = powerICA(white_data,'pow3')

          unMixed_graphpower = W_graphpower @ white_data
          unMixed_power = W_power @ white_data

          #mds_power[i] = md(W_whiten@mixmat,W_power)
          #mds_graphpower[i] = md(W_whiten@mixmat,W_graphpower)
          # mse_graphpower = MSE(unMixed_power,data.T)
          # print(mse_graphpower)
      #%%
        # print(np.mean(mds_graphpower),np.mean(mds_power))
        # fig1, axs1 = plt.subplots(1,2, sharey=True)  
        # axs1[0].boxplot(mds_power)
        # axs1[0].set_xlabel("PowerICA")
        # axs1[1].boxplot(mds_graphpower)
        # axs1[1].set_xlabel("GraphPowerICA")
        # #fig1.savefig(fname="b="+str(b)+"_nonlin="+nonlin+"_Signaltype="+Signaltype+".png")
        # plt.show()


        ### plot figures ###
        fig1, axs1 = plt.subplots(P, sharex=True)
        #plt.xlim( xmin=400, xmax=600 )
        fig2, axs2 = plt.subplots(P, sharex=True)
        #plt.xlim(xmin=400, xmax=600 )
        fig3, axs3 = plt.subplots(P, sharex=True)
        #plt.xlim(xmin=400, xmax=600 )
        fig4, axs4 = plt.subplots(P, sharex=True)
       # plt.xlim(xmin=400, xmax=600)
        fig5, axs5 = plt.subplots(P, sharex=True)
        #plt.xlim(xmin=400, xmax=600)
        i=0
        while(i<P):
            # input signals
          axs1[i].plot(data[:, i], lw=3)
          axs1[i].set_ylabel('sig: ' + str(i))
          fig1.suptitle('Input Signals')

          axs2[i].plot(mixeddata.T[:, i], lw=3)
          axs2[i].set_ylabel('sig: ' + str(i))
          fig2.suptitle('Mixed Signals')

          # axs3[i].plot(mixdata_noise.T[:, i], lw=3)
          # axs3[i].set_ylabel('sig: ' + str(i))
          # fig3.suptitle('Contaminated Mixed Signals')

          axs4[i].plot((unMixed_graphpower).T[:, i], lw=3)
          axs4[i].set_ylabel('sig: ' + str(i))
          fig4.suptitle('Recovered signals GraphPowerICA')

          axs5[i].plot((unMixed_power).T[:, i], lw=3)
          axs5[i].set_ylabel('sig: ' + str(i))
          fig5.suptitle('Recovered signals PowerICA')
          
          i = i+1
        plt.show()
# %%
