#%%
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from graph_PowerICA import*
from PowerICA import*

if __name__ == "__main__":


    data = np.stack([create_signal(f=2,c='ecg',x=1000),
                    create_signal(f=5,c='cos',x=1000),
                    create_signal(f=10,c='rect',x=1000), 
                    create_signal(f=25,c='sawt',x=1000)]).T
    
    N = data.shape[0]

    G1 = graphs.ErdosRenyi(N,p=0.5)
    G2 = graphs.ErdosRenyi(N,p=0.3)
    G3 = graphs.ErdosRenyi(N,p=0.2)
    G4 = graphs.ErdosRenyi(N,p=0.15)
    Ws = np.array([G1.W.toarray(),G1.W.toarray(),G1.W.toarray(),G1.W.toarray()]).reshape(1000,1000,4)


    mixmat = mixing_matrix(4)
    mixeddata = mixmat @ data.T
    mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=10),prop=0.00,type='impulse',std=100) for dat in mixeddata])
    white_data,W_whiten,W_dewhiten,_ = whitening(mixdata_noise, type='sample')

    W_graphpower,_ = Graph_powerICA(white_data,'pow3',Ws,0.4)
    W_power,_ = powerICA(white_data,'pow3')

    unMixed_graphpower = W_graphpower @ white_data
    unMixed_power = W_power @ white_data

    md_powerica = md(W_whiten@mixmat,W_power)
    md_graphpowerica = md(W_whiten@mixmat,W_graphpower)

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
    print(md_powerica, md_graphpowerica)
    while(i<4):
        # input signals
        axs1[i].plot(data[:, i], lw=3)
        axs1[i].set_ylabel('sig: ' + str(i))
        fig1.suptitle('Input Signals')

        axs2[i].plot(mixeddata.T[:, i], lw=3)
        axs2[i].set_ylabel('sig: ' + str(i))
        fig2.suptitle('Mixed Signals')

        axs3[i].plot(mixdata_noise.T[:, i], lw=3)
        axs3[i].set_ylabel('sig: ' + str(i))
        fig3.suptitle('Contaminated Mixed Signals')

        axs4[i].plot((unMixed_graphpower).T[:, i], lw=3)
        axs4[i].set_ylabel('sig: ' + str(i))
        fig4.suptitle('Recovered signals GraphPowerICA')

        axs5[i].plot((unMixed_power).T[:, i], lw=3)
        axs5[i].set_ylabel('sig: ' + str(i))
        fig5.suptitle('Recovered signals PowerICA')
        
        i = i+1