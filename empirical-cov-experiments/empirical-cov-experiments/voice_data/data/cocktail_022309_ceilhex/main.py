import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.linalg
import cProfile
from scipy import signal
from sklearn.decomposition import FastICA
from htica import damp
from htica import centroidOrthogonalizer

#Fs, data = scipy.io.wavfile.read('party3.wav')
data = np.loadtxt(open("../../samples/sample-9000.csv"), delimiter=",", skiprows=1)

# data is m-by-n matrix, n is number of sensors, m is number of observations
m, n = data.shape

# norm = np.sqrt(np.sum(np.power(data,2),0))
# data = data/norm

# Put in n-by-m shape
data = np.transpose(data)

# Subtract the mean
#data = np.transpose(np.transpose(data) - np.mean(data,1))

#C = (np.mat(data) * np.mat(data).T)/m
#orth = np.linalg.inv(scipy.linalg.sqrtm(C))

#data = np.mat(orth) * np.mat(data)

#R = 7
#dataDamped, dataRejected, rate = damp(data, R)
#print("Rejection rate: " + str(rate))

#ica = FastICA(fun='exp', n_components=6)
#print(ica)

#S_ = ica.fit_transform(data)
#A_ = ica.mixing_

#print(centroidOrthogonalizer(data[:,0:10000], data[:,0]))
cProfile.run('print(centroidOrthogonalizer(data))')
