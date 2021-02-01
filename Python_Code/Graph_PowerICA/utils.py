import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import mne
import os
import neurokit2 as nk
import robustsp as rsp
# from Mscat import *
from sklearn.metrics import mean_squared_error
from scipy.stats.distributions import chi2
import scipy as sp
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt


def create_signal(x=10000, c='sin', ampl=1, fs=1000, f=2, eeg_components=1):
    """
    creates a certain signal
    :param x: length of the data vector
    :param c: type of signal, e.g sin, cos, sawt, rect, ecg
    :param ampl: amplitude of signal
    :param fs: sample frequency
    :param f: desired frequency of the signal
    :return: data signal
    """
    # arange samples from 0 to x
    samples = np.arange(x)

    def sin1():
        # samples/fs = time of the signal
        # to create signal 2*pi*f*t
        s1 = np.sin(2 * np.pi * f * samples / fs)  # Signal 1 : sinusoidal signal
        return s1

    def cos1():
        s2 = np.cos(2 * np.pi * f * samples / fs)  # Signal 2 : cosinus signal
        return s2

    def sawt():
        s3 = signal.sawtooth(2 * np.pi * f * samples / fs)  # Signal 3: sawtooth signal
        return s3

    def rect():
        s4 = np.sign(np.sin(2 * np.pi * f * samples / fs))  # Signal 4: square signal
        return s4

    # def piky():
    #    s6 = ((np.arange(0,n_samples, 1) % 23)-11/9)**5  #((rem(v, 23) - 11) / 9). ^ 5 # Signal 4: square signal
    #    return s6

    def ecg():
        s7 = nk.ecg_simulate(length=x)
        return s7

    def eeg():
        if c=='eeg':
            #load sample visually/auditory evoked eeg data from mne, the whole data has 60 EEG channels 
            sample_data_folder = mne.datasets.sample.data_path()
            sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
            raw = mne.io.read_raw_fif(sample_data_raw_file)
            eeg = np.array(raw.get_data(picks='eeg')) # pick only eeg channels
            s8 = eeg[0:eeg_components, 0:x]   #return the number of channels, and samplesize as wanted
            s8 = 2 * (s8 - np.min(s8)) / (np.max(s8) - np.min(s8)) - 1  # Normalize EEG data between -1 and 1
            if s8.shape[0] == 1:
                s8 = s8.reshape(x,1)
            else:
                s8 = s8.T
            return s8

    switcher = {
        'sin': sin1(),
        'cos': cos1(),
        'sawt': sawt(),
        'rect': rect(),
        'ecg': ecg(),
        'eeg': eeg(),
        #'piky': piky(),
        'all': np.stack([cos1(), sawt(), rect(), ecg()], axis=1) #
    }

    def switch_signal(c):
        signl = switcher.get(c, "Invalid")
        return ampl * signl/np.max(signl)

    return switch_signal(c)



def apply_noise(data, type='white', SNR_dB=20, seed=None):
    """
    :param data: input signal vector
    :param type: type of noise
    :param SNR_dB: Difference between signal power and noise power in dB
    :return: data vector with applied noise
    """

    np.random.seed(seed)  # set seed for reproducible results
    data_ary = data  # np.c_()

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

    return switch_signal(type)


# apply_noise(np.linspace(0, 10, 2000), c = 'gamma', SNR_dB = 50)

def create_outlier(data, prop=0.01, std=3, type='impulse', seed=None):
    """
    :param data: input data
    :param prop: percentage of outlier dependent on datalength
    :param std: max value of outlier
    :param type: patchy or impulsive outlier
    :return: signal vector with replacement oulier
    """
    c = len(data)
    n_outl = int(prop * c)
    sigma_data = np.std(data)
    max_value = sigma_data * std

    np.random.seed(seed)
    outlier = np.random.uniform(-max_value, max_value, n_outl)

    if (type == 'impulse'):
        outlier_index = np.random.choice(a=np.arange(0, c, 1), size=n_outl)
        data[outlier_index] = outlier

    elif (type == 'patch'):
        outlier_index = np.random.choice(a=np.arange(0, c - n_outl, 1))
        data[outlier_index:outlier_index + n_outl] = outlier
    else:
        print("invalid type")

    return data


def add_artifact(data, fs,  prop=0.99, snr_dB=3, number=1, type='eye', seed=None):
    """
    Artifcats created after the types of Delorme
    inspired by:
    Enhanced detection of artifacts in EEG data using higher-order statistics and independent component analysis, 2007

    :param data:
    :param fs:
    :param prop:
    :param snr_dB:
    :param number:
    :param type:
    :param seed:
    :return:
    """
    data_outl = np.zeros_like(data)
    c = len(data)
    len_artifact = int(prop * c)
    np.random.seed(seed)

    # Calculate signal power and convert to dB
    sig_avg_watts = np.median(data ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - snr_dB
    noise_avg_watts = 10 ** (noise_avg_db / 10)


    if (type == 'eye'):
        order = 3
        lowcut = 1.0
        highcut = 10.0
        sos = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', output='sos')

        # eye_outlier = np.zeros_like(data)
        # a = np.arange(0, c - len_artifact, 1)
        # for k in range(0, number):
        #     noise = apply_noise(data, type='white', SNR_dB=snr_dB) - data
        #     noise = noise[0:len_artifact]
        #     outlier = sosfilt(sos, noise)
        #     outlier_index = np.random.choice(a)
        #     outlier_index_a = np.asarray(np.where(a == outlier_index))[0]
        #     a = np.delete(a, a[outlier_index_a-len_artifact:outlier_index_a+len_artifact])
        #     outlier = (2 * (outlier - np.min(outlier)) / (np.max(outlier) - np.min(outlier)) - 1)*noise_avg_watts  # Normalize EEG data between -1 and 1
        #     eye_outlier[outlier_index:outlier_index + len_artifact] = outlier
        #     data_outl = eye_outlier + data

        eye_outlier = np.zeros_like(data)
        a = np.arange(0, c - len_artifact, 1)
        for k in range(0, number):
            noise = apply_noise(data, type='white', SNR_dB=snr_dB) - data
            noise = noise[0:len_artifact]
            outlier = sosfilt(sos, noise)
            outlier = (2 * (outlier - np.min(outlier)) / (
                    np.max(outlier) - np.min(outlier)) - 1) * noise_avg_watts  # Normalize EEG data between -1 and
            done = False

            while done == False:
                outlier_index = np.random.choice(a)
                if outlier_index is not False:
                    #   a = np.delete(a, a[outlier_index - len_artifact:outlier_index + len_artifact])
                    outlier_indices = np.arange(start=outlier_index, stop=outlier_index + len_artifact, step=1)
                    a[(outlier_index - len_artifact):(outlier_index + len_artifact)] = False
                    eye_outlier[outlier_indices] = outlier
                    done = True
        data_outl = eye_outlier + data
        outlier = eye_outlier

    elif (type == 'muscle'):
        order = 3
        lowcut = 20.0
        highcut = 60.0
        sos = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', output='sos')

        muscle_outlier = np.zeros_like(data)
        a = np.arange(0, c - len_artifact, 1)
        for k in range(0, number):
            noise = apply_noise(data, type='white', SNR_dB=snr_dB) - data
            noise = noise[0:len_artifact]
            outlier = sosfilt(sos, noise)
            outlier = (2 * (outlier - np.min(outlier)) / (
                    np.max(outlier) - np.min(outlier)) - 1) * noise_avg_watts  # Normalize EEG data between -1 and
            done = False

            while done == False:
                outlier_index = np.random.choice(a)
                if outlier_index is not False:
                #   a = np.delete(a, a[outlier_index - len_artifact:outlier_index + len_artifact])
                    outlier_indices = np.arange(start=outlier_index, stop=outlier_index+len_artifact, step=1)
                    a[(outlier_index - len_artifact):outlier_index + len_artifact] = False
                    muscle_outlier[outlier_indices] = outlier
                    done = True

        data_outl = muscle_outlier + data
        outlier = muscle_outlier

    elif (type == 'linear'):
        linear_outlier = np.zeros_like(data)
        a = np.arange(0, c - len_artifact, 1)
        outlier = np.arange(start=-1, stop=1, step=2 / len_artifact) * noise_avg_watts
        for k in range(0, number):

            if number == 2:
                noise_avg_watts = noise_avg_watts*0.5
            done = False

            while done == False:
                outlier_index = np.random.choice(a)
                if outlier_index is not False:
                    outlier_indices = np.arange(start=outlier_index, stop=outlier_index+len_artifact, step=1)
                    # a = np.delete(a, a[outlier_index - len_artifact:outlier_index + len_artifact])
                    a[(outlier_index - len_artifact):outlier_index + len_artifact] = False
                    linear_outlier[outlier_indices] = outlier
                    done = True
        data_outl = linear_outlier + data
        outlier = linear_outlier

    elif (type == 'electric'):
        electric_outlier = np.zeros_like(data)
        outlier = np.ones_like(data) * noise_avg_watts
        a = np.arange(0, c - len_artifact, 1)
        for k in range(0, number):

            if number == 2:
                len_artifact_e = int(len_artifact/4)
            else:
                len_artifact_e = len_artifact
            done = False

            while done == False:
                outlier_index = np.random.choice(a)
                if outlier_index is not False:
                    outlier_indices = np.arange(start=outlier_index, stop=outlier_index + len_artifact_e, step=1)
                    a[(outlier_index - len_artifact_e):outlier_index + len_artifact_e] = False
                    electric_outlier[outlier_indices] = outlier[outlier_indices]
                    done = True

        data_outl = electric_outlier + data
        outlier = electric_outlier

    elif (type == 'noise'):
        noise_outlier = np.zeros_like(data)
        outlier = (apply_noise(data, type='white', SNR_dB=snr_dB) - data)
        outlier = (outlier/max(outlier)) * noise_avg_watts
        a = np.arange(0, c - len_artifact, 1)
        for k in range(0, number):
            done = False

            while done == False:
                outlier_index = np.random.choice(a)
                if outlier_index is not False:
                    outlier_indices = np.arange(start=outlier_index, stop=outlier_index + len_artifact, step=1)
                    a[(outlier_index - len_artifact):outlier_index + len_artifact] = False
                    noise_outlier[outlier_indices] = outlier[outlier_indices]
                    done = True

        data_outl = noise_outlier + data
        outlier = noise_outlier
    else:
        print("invalid type")

    return data_outl, outlier


def mixing_matrix(n_components, seed=None, m=0):
    """Creates random mixing Ma

    Args:
        n_components (integer): Number of source signals.
        seed (integer): Set seed for reproducibility.If None mixing matrix will be different each time.

    Returns:
        mixingmat[array]: Matrix for linear mixing of n source signals.
    """
    np.random.seed(seed)
    #if seed is not None:
    #   np.random.seed(seed)
    n_components_x = n_components + m
    n_components_y = n_components
    mixingmat = np.random.rand(n_components_x, n_components_y)

    return mixingmat


def whitening(x, type='sample', loss='Huber', percentile=1, r=0):
    """linearly transform the observed signals X in a way that potential 
       correlations between the signals are removed and their variances equal unity. 
       As a result the covariance matrix of the whitened signals will be equal to the identity matrix

    Args:
        x (array): Centered input data.
        type (string): type of covariance estimation
        loss (string): type of M-estimator for loss in Mscat
        percentile: percentile of total variance to be represented by the components (if 1 returns all principal components)

    Returns:
        x_whitened: Whitened Data with covariance matrix that is equal to identity matrix.
    """
    centered_X, _ = centering(x)
    # plt.plot(centered_X.T)
    # plt.show()

    cov = covariance(centered_X, type, loss) # covariance(centered_X) #calculate Covariance matrix between signals
    d, E = np.linalg.eig(cov)#Eigenvalue decomposition (alternatively one can use SVD for faster computation)
    # e1, e2 = E.shape
    # #print(d)
    # d = np.real(d)
    #     E = np.real(E.flatten()).reshape(e1, e2)
    idx = d.argsort()[::-1]
    #Sort eigenvalues  
    cumul_var = np.cumsum(d)
    N = len(d)
    n_components=N
    if percentile < 1:
        for i in range(len(cumul_var)):
            if cumul_var[i]/cumul_var[N-1] > percentile: # check at what index the cumulated variance contains p (percentile) of the total variance 
                n_components = i+1
                break

        #print(n_components)   
        d = d[idx[0:n_components]]                #Sort eigenvalues
        E = E[:,idx[0:n_components]]              #Sort eigenvectors

        D_inv = np.diag(1/np.sqrt(d+r))   #Calculate D^(-0.5)
        W_whiten =  D_inv @ E.T         #Calculate Whitening matrix W_whiten 
        #W_dewhiten = E @ np.diag(np.sqrt(d))
        x_whitened = (W_whiten @ centered_X)
        return x_whitened, W_whiten, n_components

    elif percentile==1:   
        D_inv = np.diag(1/np.sqrt(d+r))   #Calculate D^(-0.5)
        W_whiten =  D_inv @ E.T         #Calculate Whitening matrix W_whiten 
        W_dewhiten = E @ np.diag(np.sqrt(d+r))
        x_whitened = (W_whiten @ centered_X)
        return x_whitened, W_whiten, W_dewhiten, n_components


def centering(x, type='sample'):
    """Centers input data x by subtracting the mean of each Signal represented by the row vectors of x.

    Args:
        x ([type]): [description]
        type: sample mean or spatial median
    Returns:
        centered [array]: Centered data array.
        mean [array]: Vector containing the mean of each signal.
    """
    # mean = np.mean(x, axis=1, keepdims=True)
    ary = np.copy(x)
    if(type == 'sample'):
        mean = np.apply_along_axis(np.mean, axis=1, arr=ary)
    if(type == 'spatmed'):
        mean = spatmed(ary.T)
    centered = ary
    n, m = np.shape(ary)
    for i in range(0, n, 1):
        centered[i, :] = centered[i, :] - mean[i]
    # print(centered)
    return centered, mean


def covariance(x, type='sample', loss=None):
    """Calculate Covariance matrix for the rowvectors of the input data x.

    Args:
        x (array): Mixed input data.
        type: type of cov estimation

    Returns:
        cov [array]: Covariance Matrix of the signals represented by the rowvectors of x.
    """

    cov_signcm = 1
    cov_Mscat = Mscat(x.T, loss='Huber')
    n = np.shape(x)[1] - 1

    if (type == 'sample'):
        mean = np.mean(x, axis=1, keepdims=True)
        m = x - mean
        cov_sample = m @ m.T / n
        cov = cov_sample
    elif (type == 'signcm'):
        cov = cov_signcm
    elif (type == 'spatmed'):
        spatmed1 = spatmed(x.T)
        m = x - np.expand_dims(spatmed1, axis=1)
        cov_spatmed = m @ m.T / n
        cov = cov_spatmed
    elif(type == 'Mscat'):
        [cov, _, _, _] = cov_Mscat

    return cov


def Mscat(X, loss, losspar=None, invCx=None, printitn=0, MAX_ITER=1000, EPS=1.0e-5):
    """
    [C,invC,iter,flag] = Mscat(X,loss,...)
    computes M-estimator of scatter matrix for the n x p data matrix X
    using the loss function 'Huber' or 't-loss' and for a given parameter of
    the loss function (i.e., q for Huber's or degrees of freedom v for
    the t-distribution).

    Data is assumed to be centered (or the symmetry center parameter = 0)

    INPUT:
           X: the data matrix with n rows (observations) and p columns.
        loss: either 'Huber' or 't-loss' or 'Tyler'
     losspar: parameter of the loss function: q in [0,1) for Huber and
              d.o.f. v >= 0 for t-loss. For Tyler you do not need to specify
              this value. Parameter q determines the treshold
              c^2 as the qth quantile of chi-squared distribution with p
              degrees of freedom distribution (Default q = 0.8). Parameter v
              is the def.freedom of t-distribution (Default v = 3)
              if v = 0, then one computes Tyler's M-estimator
        invC: initial estimate is the inverse scatter matrix (default =
              inverse of the sample covariance matrix)
    printitn: print iteration number (default = 0, no printing)
    OUTPUT:
           C: the M-estimate of scatter using Huber's weights
        invC: the inverse of C
        iter: nr of iterations
        flag: flag (true/false) for convergence
    """
    def tloss_consistency_factor(p, v):
        '''
        computes the concistency factor b = (1/p) E[|| x ||^2 u_v( ||x||^2)] when
        x ~N_p(0,I).
        '''
        sfun = lambda x: (x ** (p / 2) / (v + x) * np.exp(-x / 2))
        c = 2 ** (p / 2) * sp.special.gamma(p / 2)
        q = (1 / c) * \
            sp.integrate.quad(sfun, 0, np.inf)[0]
        return ((v + p) / p) * q  # consistency factor

    X = np.asarray(X);
    n, p = X.shape
    realdata = np.isrealobj(X)

    # SCM initial start
    invC = np.linalg.pinv(X.conj().T @ X / n) if invCx == None else np.copy(invCx)

    if loss == 'Huber':
        ufun = lambda t, c: ((t <= c) + (c / t) * (t > c))  # weight function u(t)
        q = 0.9 if losspar == None else losspar
        if np.isreal(q) and np.isfinite(q) and 0 < q and q < 1:
            if realdata:
                upar = chi2.ppf(q, df=p)  # threshold for Huber's weight u(t;.)
                b = chi2.cdf(upar, p + 2) + (upar / p) * (1 - q)  # consistency factor
            else:
                upar = chi2.ppf(q, 2 * p) / 2
                b = chi2.cdf(2 * upar, 2 * (p + 1)) + (upar / p) * (1 - q)
        else:
            raise ValueError('losspar is a real number in [0,1] and not %s for Huber-loss' % q)
        const = 1 / (b * n)
    if loss == 't-loss':
        # d.o.f v=3 is used as the default parameter for t-loss
        # otherwise use d.o.f. v that was given
        upar = 3 if losspar == None else losspar
        if not np.isreal(upar) or not np.isfinite(upar) or upar < 0:
            raise ValueError('losspar should be a real number greater 0 and not %s for t-loss' % q)
        if realdata and upar != 0:
            # this is for real data
            ufun = lambda t, v: 1 / (v + t)  # weight function
            b = tloss_consistency_factor(p, upar)
            const = (upar + p) / (b * n)
        if not realdata and upar != 0:
            # this is for complex data
            ufun = lambda t, v: 1 / (v + 2 * t)  # weight function
            b = tloss_consistency_factor(2 * p, upar)
            const = (upar + 2 * p) / (b * n)
        if upar == 0:
            # Tylers M-estimator
            ufun = lambda t, v: 1 / t
            const = p / n

    for i in range(MAX_ITER):
        t = np.real(np.sum((X @ invC) * np.conj(X), axis=1))  # norms
        C = const * X.conj().T @ (X * ufun(t, upar)[:, None])
        d = np.max(np.sum(np.abs(np.eye(p) - invC @ C), axis=1))

        if printitn > 0 and (i + 1) % printitn == 0:
            print("At iter = %d, dis=%.6f\n" % (i, d))
        invC = np.linalg.pinv(C)
        if d <= EPS: break

    if i == MAX_ITER: print("WARNING! Slow convergence: the error of the solution is %f\n'" % d)
    return C, invC, i, i == MAX_ITER - 1


def spatmed(X, printitn=0, iterMAX=500, EPS=1e-6, TOL=1e-5):
    '''
      Computes the spatial median based on (real or complex) data matrix X.
      INPUT:
             X: Numeric data matrix of size N x p. Each row represents one
               observation, and each column represents one variable
     printitn : print iteration number (default = 0, no printing)

     OUTPUT
          smed: Spatial median estimate
    '''
    l = np.sum(X * np.conj(X), axis=1)
    X = X[l != 0, :]
    n = len(X)

    smed0 = np.median(X) if np.isrealobj(X) else np.mean(X)
    norm0 = np.linalg.norm(smed0)

    for it in range(iterMAX):
        Xc = X - smed0
        l = np.sqrt(np.sum(Xc * np.conj(Xc), axis=1, keepdims=1)) # modified
        l[l < EPS] = EPS
        Xpsi = np.divide(Xc, l)  # np.expand_dims(l, axis=1) Xc / l
        update = np.sum(Xpsi, axis=0) / sum(1 / l)
        smed = smed0 + update

        dis = np.linalg.norm(update, ord=2) / norm0

        if printitn > 0 and (it + 1) % printitn == 0: print('At iter = %.3d, dis =%.7f \n' % (it, dis))

        if dis <= TOL: break
        smed0 = smed
        norm0 = np.linalg.norm(smed, ord=2)
    return smed


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
    - https://github.com/sweichwald/coroICA-python
    """

    # first dimensions of A (true Mixing Matrix)
    d = np.shape(A)[0]
    #calculate Gain Marix
    G = Vhat.dot(A)
    # transform into maximization problem and calculate new gain
    Gsq = np.abs(G)**2
    Gtilde = Gsq / (Gsq.sum(axis=1)).reshape((d, 1))
    # Define the maximization problem
    costmat = 1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=1), d).reshape((d, d))

    row_ind, col_ind = linear_sum_assignment(costmat)
    md = np.sqrt(d - np.sum(np.diag(Gtilde[row_ind, col_ind]))) / \
        np.sqrt(d - 1)
    return md


def MSE(pred_signals, true_signals):
    """
      Computes the MSE between the estimated and true signals

      INPUT:
            pred_signals: Array of size (dim,n) containing the estimated signals as row vectors
            true_signals: Array of size (dim,n) containing the true signals as row vectors
      OUTPUT:
            MSE: Array of size length dim containing the MSE of each estimated signal corresponding to the true Signal
    """

    dim, n = np.shape(true_signals)
    pred_mean = np.mean(pred_signals,axis=1).reshape(dim, 1)
    pred_std = np.std(pred_signals,axis=1).reshape(dim, 1)
    true_mean = np.mean(true_signals,axis=1).reshape(dim, 1)
    true_std = np.std(true_signals,axis=1).reshape(dim, 1)
    #scale data to have unit variance 1/n*y@y.T=1
    pred_signals = np.divide((pred_signals - pred_mean), pred_std)
    true_signals = np.divide((true_signals - true_mean), true_std)
    MSE = np.zeros(dim)
    MSE_matrix1 = np.zeros((dim, dim))
    MSE_matrix2 = np.zeros((dim, dim))

    #calculate MSE between all estimated signals with changed sign and true signals and store in array
    for i in range(0, dim):
        for j in range(0, dim):
            MSE_matrix1[i, j] = mean_squared_error(true_signals[j],pred_signals[i])
    #calculate MSE between all estimated and true signals  and store in array
    for i in range(0, dim):
        for j in range(0, dim):
            MSE_matrix2[i, j] = mean_squared_error(true_signals[j],-pred_signals[i])
    
    #find minima of all possible MSE values
    array_min1 = MSE_matrix1.min(axis=1)
    array_min2 = MSE_matrix2.min(axis=1)

    #store minima in separate array
    for i in range(dim):
        if(array_min1[i]>array_min2[i]):
            MSE[i] = array_min2[i]
        else:
            MSE[i] = array_min1[i]

    return MSE


def SNR(pred_signals, true_signals):
    """[summary]

    Args:
        pred_signals (dim,n): Array of size (dim,n) containing the estimated signals as row vectors
        true_signals (dim,n): Array of size (dim,n) containing the true signals as row vectors

    Returns:
        SNR(dim): Array of length dim containing the SNR of each estimated signal corresponding to the true Signal in dB
    """
    dim, n = np.shape(true_signals)
    pred_mean = np.mean(pred_signals, axis=1).reshape(dim, 1)
    pred_std = np.std(pred_signals, axis=1).reshape(dim, 1)
    true_mean = np.mean(true_signals, axis=1).reshape(dim, 1)
    true_std = np.std(true_signals, axis=1).reshape(dim, 1)
    #scale data to have unit variance 1/n*y@y.T=1
    pred_signals = np.divide((pred_signals - pred_mean), pred_std)
    true_signals = np.divide((true_signals - true_mean), true_std)
    SNR = np.zeros(dim)
    SNR_matrix1 = np.zeros((dim, dim))
    SNR_matrix2 = np.zeros((dim, dim))

    #calculate SNR between all estimated and true signals and store in array
    for i in range(0,dim):
        for j in range(0,dim):
            pred_var1 = np.var(pred_signals[i])
            diff_var1 = np.var(pred_signals[i]-true_signals[j])
            SNR_matrix1[i,j] = pred_var1/diff_var1
    #calculate SNR between all estimated and true signals with flipped sign and store in array
    for i in range(0, dim):
        for j in range(0, dim):
            pred_var2 = np.var(pred_signals[i])
            diff_var2 = np.var(pred_signals[i]+true_signals[j])
            SNR_matrix2[i, j] = pred_var2/diff_var2
   
    #find maxima of all possible SNR values
    array_max1 = SNR_matrix1.max(axis=1)
    array_max2 = SNR_matrix2.max(axis=1)
    
    #store maxima in separate array
    for i in range(dim):
        if(array_max1[i]>array_max2[i]):
            SNR[i]=array_max1[i]
        else:
            SNR[i]=array_max2[i]

    return 10*np.log10(SNR)
