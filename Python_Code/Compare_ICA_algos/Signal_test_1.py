#%%
import numpy as np
import matplotlib.pyplot as plt
import PowerICA
import os
import sys
from coroica import CoroICA
from icecream import ic
import robustsp as rsp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'Python_Code','JADE'))
sys.path.append(os.path.join(BASE_DIR,'utils'))
# print(os.path)
# print(BASE_DIR)
from utils import *
from jade import jadeR
#from monte_carlo_random_walk import monte_carlo_run
from distances import *
from fast_Radical import *
# create example signals:
data = np.stack([create_signal(f= 2, c='ecg'),
                 create_signal(f = 5, ampl=4,c='cos'),
                 create_signal(f = 10, c='rect'),
                 create_signal(f = 25,c='sawt')]).T

# data_premixing_contamination = data
# data_premixing_contamination[2]= create_outlier(data_premixing_contamination[2])

# Standardize data
#data /= data.std(axis=0)  # Standardize data
#%%
# create mixing matrix and mixed signals
c, r = data.shape
MM = mixing_matrix(r,seed=1)
mixdata = MM@data.T

#ic(MM)
#apply noise
#mixdata_noise = np.stack([create_outlier(apply_noise(dat,type='white', SNR_dB=20),prop=0.001,std=5) for dat in mixdata])
mixdata_noise = mixdata
#%%
# centering the data and whitening the data:
white_data,W_whiten,W_dewhiten,_ = whitening(mixdata_noise, type='sample')

# perform PowerICA
W_power, _ = PowerICA.powerICA(white_data, 'pow3')

#W_radical = RADICAL(X_data)

# perform Jade
W_jade = jadeR(white_data, rho_x= lambda x: np.median(x,axis = 1), robust_loc= True, verbose= False)
W_jade = np.squeeze(np.asarray(W_jade))

#Perform fastICA
W_radical = RADICAL(white_data)
#W_fast = sklearn.decomposition.FastICA(n_components=4,whiten=True)
#print(mixdata_noise.shape)
c = CoroICA(partitionsize = 10 ,groupsize= 10000)
c.fit(white_data.T)
W_coro = c.V_
#%%
# Un-mix signals using
unMixed_power = W_power @ white_data
unMixed_jade = W_jade @ white_data
unMixed_radical = W_radical @ white_data
unMixed_coro = W_coro @ white_data

#Compute Minimum Distance Index of different ICA algorithms

#md_radical = md(MM,np.linalg.inv((W_dewhiten@ W_radical.T)))
#md_powerica = md(MM,np.linalg.inv((W_dewhiten@ W_power.T)))
#md_jade = md(MM,np.linalg.inv((W_dewhiten@ W_jade.T)))
#md_random = md(MM, np.random.randn(r,r))
md_radical = md(W_whiten@MM,W_radical)
md_powerica = md(W_whiten@MM,W_power)
md_jade = md(W_whiten@MM,W_jade)
md_coro = md(W_whiten@MM,W_coro)
np.random.seed(20)
md_random = md(MM, np.random.randn(r,r))

print('Minimum Distances Index\n Jade:',md_jade,'\n','Radical:',md_radical,'\n','PowerICA:',md_powerica,'\n', 'CoroICA:',md_coro,'\n''Random:',md_random)


# #%%
# #unMixed_fast= W_fast @ X_data
# # Plot input signals (not mixed)
# i = 0
# fig1, axs1 = plt.subplots(r, sharex=True)
# fig2, axs2 = plt.subplots(r, sharex=True)
# fig3, axs3 = plt.subplots(r, sharex=True)
# fig4, axs4 = plt.subplots(r, sharex=True)
# fig5, axs5 = plt.subplots(r, sharex=True)
# fig6, axs6 = plt.subplots(r, sharex=True)
# fig7, axs7 = plt.subplots(r, sharex=True)

# while(i<r):
#     # input signals
#     axs1[i].plot(data[:, i], lw=3)
#     axs1[i].set_ylabel('sig: ' + str(i))
#     fig1.suptitle('Input Signals')


#     axs2[i].plot(mixdata.T[:, i], lw=3)
#     axs2[i].set_ylabel('sig: ' + str(i))
#     fig2.suptitle('Mixed Signals')

#     axs3[i].plot(mixdata_noise.T[:, i], lw=3)
#     axs3[i].set_ylabel('sig: ' + str(i))
#     fig3.suptitle('Contaminated Mixed Signals')

#     axs4[i].plot(unMixed_power.T[:, i], lw=3)
#     axs4[i].set_ylabel('sig: ' + str(i))
#     fig4.suptitle('Recovered signals PowerICA')

#     axs5[i].plot(unMixed_jade.T[:, i], lw=3)
#     axs5[i].set_ylabel('sig: ' + str(i))
#     fig5.suptitle('Recovered signals JADE')

#     axs6[i].plot(unMixed_radical.T[:, i], lw=3)
#     axs6[i].set_ylabel('sig: ' + str(i))
#     fig6.suptitle('Recovered signals RADICAL')

#     axs7[i].plot(unMixed_coro.T[:, i], lw=3)
#     axs7[i].set_ylabel('sig: ' + str(i))
#     fig7.suptitle('Recovered signals CoroICA')

#     i = i+1

# plt.show()

# # '''
# # ax1 = fig1.add_subplot(1, i, figsize=[18, 5])
# # ax.plot(data, lw=3)
# # ax.tick_params(labelsize=12)
# # ax.set_xticks([])
# # ax.set_yticks([-1, 1])
# # ax.set_title('Source signals', fontsize=25)
# # #ax.set_xlim(0, 100)

# # fig, ax = plt.subplots(1, 1, figsize=[18, 5])
# # ax.plot(mixdata.T, lw=3)
# # ax.tick_params(labelsize=12)
# # ax.set_xticks([])
# # ax.set_yticks([-1, 1])
# # ax.set_title('Mixed signals', fontsize=25)
# # #ax.set_xlim(0, 100)

# # fig, ax = plt.subplots(1, 1, figsize=[18, 5])
# # ax.plot(unMixed.T, label='Recovered signals', lw=3)
# # ax.set_xlabel('Sample number', fontsize=20)
# # ax.set_title('Recovered signals', fontsize=25)
# # #ax.set_xlim(0, 100)

# # plt.show()
# # '''



# # %%

# %%
