import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import neurokit2 as nk
import robustsp as rsp

from sklearn.decomposition import FastICA, PCA
from scipy.optimize import linear_sum_assignment

def create_signal(x = 2000, c = 'sin', ampl = 1, fs = 2):
    """
    creates a certain signal
    :param x: length of the data vector
    :param c: type of signal, e.g sin, cos, sawt, rect, ecg
    :param ampl: amplitude of signal
    :param fs: sample frequency
    :return: data signal
    """
    #
    # x is number of samples
    # sample rate
    n_samples = x
    time = np.linspace(0, 10, n_samples)

    def sin1():
        s1 = np.sin(fs * time)  # Signal 1 : sinusoidal signal
        return s1

    def cos1():
        s2 = np.cos(fs * time)  # Signal 2 : cosinus signal
        return s2

    def sawt():
        s3 = signal.sawtooth(fs * np.pi * time)  # Signal 3: sawtooth signal
        return s3

    def rect():
        s4 = np.sign(np.sin(fs * time))  # Signal 4: square signal
        return s4

    #def piky():
    #    s6 = ((np.arange(0,n_samples, 1) % 23)-11/9)**5  #((rem(v, 23) - 11) / 9). ^ 5 # Signal 4: square signal
    #    return s6

    def ecg():
        s7 = nk.ecg_simulate(length=n_samples)
        return s7

    switcher = {
        'sin': sin1(),
        'cos': cos1(),
        'sawt': sawt(),
        'rect': rect(),
        'ecg': ecg(),
        #'piky': piky(),
        'all': np.stack([sin1(), cos1(), sawt(), rect(), ecg()], axis=1) #
    }

    def switch_signal(c):
        signl = switcher.get(c, "Invalid")
        return ampl * signl

    return switch_signal(c)



def apply_noise(data, c = 'white', SNR_dB = 20):
    """
    :param data: input signal vector
    :param c: type of noise
    :param SNR_dB: Difference bewteen signal power and noise power in dB
    :return: data vector with applied noise
    """

    np.random.seed(1)  # set seed for reproducible results
    data_ary = data    #np.c_()

    data_power = data_ary ** 2
    # Set a target SNR
    target_snr_db = SNR_dB
    # Calculate signal power and convert to dB
    sig_avg_watts = np.median(data_power)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    def make_whiteNoisedB(data, target_SNR_dB):

        mean_noise = 0
        # Generate noise samples
        noise_dB = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), size=data_ary.shape)

        # Noise up the original signal (again) and plot
        data_noise = data_ary + noise_dB

        return data_noise

    def make_laplaceNoisedB(data, target_SNR_dB):

        mean_noise = 0
        noise_dB = np.random.laplace(mean_noise, np.sqrt(noise_avg_watts), size=data_ary.shape)

        # Noise up the original signal (again) and plot
        data_noise = data_ary + noise_dB

        return data_noise

    def make_gammaNoisedB(data, target_SNR_dB):

        noise_dB = np.random.gamma(shape=1, scale=np.sqrt(noise_avg_watts), size=data_ary.shape)

        #plt.plot(noise_dB)
        #plt.show()
        # Noise up the original signal (again) and plot
        data_noise = data_ary + noise_dB

        return data_noise

    def make_uniformNoisedB(data, target_SNR_dB):

        noise_dB = np.random.uniform(-np.sqrt(noise_avg_watts), np.sqrt(noise_avg_watts), size=data_ary.shape)

        # Noise up the original signal (again) and plot
        data_noise = data_ary + noise_dB

        return data_noise

    def make_expNoisedB(data, target_SNR_dB):

        noise_dB = np.random.exponential(np.sqrt(noise_avg_watts), size=data_ary.shape)

        # Noise up the original signal (again) and plot
        data_noise = data_ary + noise_dB

        return data_noise

    switcher = {
        'white': make_whiteNoisedB(data, SNR_dB),
        'laplace': make_laplaceNoisedB(data, SNR_dB),
        'gamma': make_gammaNoisedB(data, SNR_dB),
        'uniform': make_uniformNoisedB(data, SNR_dB),
        'exp': make_expNoisedB(data, SNR_dB),
    }

    def switch_signal(c):
        signl_noise = switcher.get(c, "Invalid")
        return signl_noise

    return switch_signal(c)


#apply_noise(np.linspace(0, 10, 2000), c = 'gamma', SNR_dB = 50)

def create_outlier(data, prop = 0.01, std = 3, type = 'impulse', seed = 1):
    """
    :param data: input data
    :param prop: percentage of outlier dependent on datalength
    :param std: max value of outlier
    :param type: patchy or impulsive outlier
    :return: signal vector with replacement oulier
    """
    c = len(data)
    n_outl = int(prop*c)
    sigma_data = np.std(data)
    max_value = sigma_data*std

    np.random.seed(seed)
    outlier = np.random.uniform(-max_value, max_value, n_outl)

    if(type == 'impulse'):
        outlier_index = np.random.choice(a=np.arange(0,c,1), size=n_outl)
        data[outlier_index] = outlier

    elif(type == 'patch'):
        outlier_index = np.random.choice(a=np.arange(0, c-n_outl, 1))
        data[outlier_index:outlier_index+n_outl] = outlier
    else:
        print("invalid type")

    return data

#data = np.sin(np.arange(0,200, 0.1) * 0.25 )
#data_out = create_outlier(data, prop=0.1, std=5, type='impulse')
#plt.plot(data_out)
#plt.show()

def mixing_matrix(n_components,seed = 1):
    """Creates random mixing Ma

    Args:
        n_components (integer): Number of source signals.
        seed (integer): Set seed for reproducibility.If None mixing matrix will be different each time.

    Returns:
        mixingmat[array]: Matrix for linear mixing of n source signals.
    """
    if seed is not None:
        np.random.seed(seed)
    mixingmat = np.random.rand(n_components, n_components)

    return mixingmat


def whitening(x, type='sample'):
    """linearly transform the observed signals X in a way that potential 
       correlations between the signals are removed and their variances equal unity. 
       As a result the covariance matrix of the whitened signals will be equal to the identity matrix

    Args:
        x (array): Centered input data.
        type (string): type of covariance estimation

    Returns:
        x_whitened: Whitened Data with covariance matrix that is equal to identity matrix.
    """
    centered_X, _ = centering(x)

    cov = np.cov(centered_X, bias=True) # covariance(centered_X) #calculate Covariance matrix between signals

    d, E = np.linalg.eig(cov) #Eigenvalue decomposition (alternatively one can use SVD for whitening)
    idx = d.argsort()[::-1]   
    d = d[idx]                #Sort eigenvalues
    E = E[:,idx]              #Sort eigenvectors

    D_inv = np.diag(1/np.sqrt(d))   #Calculate D^(-0.5)

    W_whiten =  D_inv @ E.T         #Calculate Whitening matrix W_whiten 
    x_whitened = (W_whiten @ centered_X)

    return x_whitened


def centering(x):
    """Centers input data x by subtracting the mean of each Signal represented by the row vectors of x.

    Args:
        x ([type]): [description]

    Returns:
        centered [array]: Centered data array.
        mean [array]: Vector containing the mean of each signal.
    """
    #mean = np.mean(x, axis=1, keepdims=True)
    mean = np.apply_along_axis(np.mean,axis=1,arr=x)
    centered = x
    n,m = np.shape(x)
    for i in range(0,n,1):
        centered[i,:] = centered[i,:]-mean[i]
    #print(centered)
    return centered, mean

def covariance(x, type='sample', loss=None):
    """Calculate Covariance matrix for the rowvectors of the input data x.

    Args:
        x (array): Mixed input data.
        type: type of cov estimation

    Returns:
        cov [array]: Covariance Matrix of the signals represented by the rowvectors of x.
    """

    cov_signcm = 1 #rsp.Covariance.signcm(x)
    cov_spatmed = 1 #rsp.Covariance.spatmed(x)
    cov_Mscat = 1 #rsp.Covariance.Mscat(x, loss=loss)

    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean
    cov_sample = m@m.T/n

    if(type == 'sample'):
        cov = cov_sample
    elif(type == 'signcm'):
        cov = cov_signcm
    elif (type == 'spatmed'):
        cov = cov_spatmed
    else:
        [cov, _, _, _] = cov_Mscat

    return cov


def md(A, Vhat):
    """Minimum distance index as defined in
    P. Ilmonen, K. Nordhausen, H. Oja, and E. Ollila.
    A new performance index for ICA: Properties, computation and asymptotic
    analysis.
    In Latent Variable Analysis and Signal Separation, pages 229–236. Springer,
    2010.

    This Code is from the coroICA package which implements the coroICA algorithm presented in 
    "Robustifying Independent Component Analysis by Adjusting for Group-Wise Stationary Noise" 
    by N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf.
    - https://github.com/sweichwald/coroICA-python"""
    # first dimensions of A (true Mixing Matrix)
    d = np.shape(A)[0]
    #calculate Gain Marix
    G = (Vhat).dot(A)
    # transform into maximization problem and calculate new gain
    Gsq = np.abs(G)**2
    temp = Gsq.sum(axis=1)
    # reshape that calculation works
    temp = temp.reshape((d, 1))
    Gtilde = Gsq / temp                #(Gsq.sum(axis=1)).reshape((d, 1))
    # Define the maximization problem
    costmat = 1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=1), d).reshape((d, d))

    row_ind, col_ind = linear_sum_assignment(costmat)
    P = Gtilde[row_ind, col_ind]
    md = np.sqrt(d - np.sum(np.diag(P))) / np.sqrt(d - 1)
    return md