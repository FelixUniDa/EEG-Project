#%%
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from graph_PowerICA import*
from PowerICA import*

if __name__ == "__main__":

    N = 1000
    P = 4
    ### Set up experiment as in R Code ###
    ### Erdos Renyi Graph ###
    G1 = graphs.ErdosRenyi(N,p=0.05).W.toarray()
    G2 = G1 @ G1
    Ws = np.array([G1,G1,G1,G1,G2,G2,G2,G2]).reshape(1000,1000,8)

    ### Graph Moving Average Signals ###
    e = np.zeros((N,P))
    e[:,0] = np.random.standard_t(5,N)
    e[:,1] = np.random.standard_t(10,N)
    e[:,2] = np.random.standard_t(15,N)
    e[:,3] = np.random.randn(N)

    theta = np.array([0.02,0.04,0.06,0.08])
    S = np.zeros((N, P))
    for j in range(0,P):
      S[:,j] = e[:,j]+theta[j]*Ws[:,:,j]@e[:,j]   

    ### Mix Signals and whiten ###
    mixmat = mixing_matrix(4)
    mixeddata = mixmat @ S.T
    #mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=10),prop=0.00,type='impulse',std=100) for dat in mixeddata])
    white_data,W_whiten,W_dewhiten,_ = whitening(mixeddata, type='sample')

    W_graphpower,_ = Graph_powerICA(white_data,'pow3',Ws,b=1)
    W_power,_ = powerICA(white_data,'pow3')

    unMixed_graphpower = W_graphpower @ white_data
    unMixed_power = W_power @ white_data

    md_powerica = md(W_whiten@mixmat,W_power)
    md_graphpowerica = md(W_whiten@mixmat,W_graphpower)
    print(md_powerica, md_graphpowerica)

    ### plot figures ###
    fig1, axs1 = plt.subplots(4, sharex=True)
    plt.xlim(xmin=0, xmax=1000 )
    fig2, axs2 = plt.subplots(4, sharex=True)
    plt.xlim(xmin=0, xmax=1000 )
    fig3, axs3 = plt.subplots(4, sharex=True)
    plt.xlim(xmin=0, xmax=1000 )
    fig4, axs4 = plt.subplots(4, sharex=True)
    plt.xlim(xmin=0, xmax=1000)
    fig5, axs5 = plt.subplots(4, sharex=True)
    plt.xlim(xmin=0, xmax=1000)
    i=0
    while(i<4):
        # input signals
        axs1[i].plot(S[:, i], lw=3)
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