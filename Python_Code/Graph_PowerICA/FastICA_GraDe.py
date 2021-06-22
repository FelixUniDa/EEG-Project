#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pygsp import graphs
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils', 'utils'))

from utils.utils import *


def GraphAutoCorrelation(X, Ws):
  n = np.shape(X)[0]
  K = np.shape(X)[1]
  P = np.shape(Ws)[2]

  ### calculate graph autocovariance Stilde ###
  Stilde = np.zeros((K, K, P))
  for p in range(0, P):
    Yw = Ws[:, :, p] @ X
    Yw = Yw / np.sqrt(np.mean(Yw ** 2, axis=0))
    Stilde[:, :, p] = (X.T @ Yw) / n
    Stilde[:, :, p] = (Stilde[:, :, p] + Stilde[:, :, p].T) / 2

  return Stilde, P


def FastICA_GraDe(X, Ws, nonlin = 'tanh', b=0.3, eps=1e-06, maxiter=1000):
  """BSS Algorithm that combines Graph decorrelation with FastICA. 
     It implements the Algorithm as given in 
     "Graph Signal Processing Meets Blind Source Separation", Submitted on 24 Aug 2020, 
     Jari Miettinen, Eyal Nitzan, Sergiy A. Vorobyov, Esa Ollila

  Args:
      X (n , K): Array containing data in column vectors
      Ws (n , n , P): Adacency \ Weight Matrices
      G (String): nonlinearity (unnecessary for PowerICA)
      g (String): first derivative of nonlin
      dg (String): second derivative of nonlin
      b (int): weight parameter (aims to balance objective function) 
      eps (float, optional): Minimum tolerance for algorithm termination. Defaults to 1e-06.
      maxiter (int, optional): Maximum number of Iterations Defaults to 1000.

  Returns:
      GAMMA (K , K): Estimate of unmixing matrix
  """
  n = np.shape(X)[0]  # number of samples
  K = np.shape(X)[1]  # number of signals
  P = np.shape(Ws)[2] # number of graphs

  ###           prewhitening            ###
  ### We can use our whitening function ###
  # mu = np.mean(X, axis=0)
  # cov = np.cov(X, rowvar=False)
  # d,E = np.linalg.eig(cov)
  # W_Whitening = E @ np.diag(d**(-0.5))@ E.T
  # X_centered = X - mu
  # Y = X_centered @ W_Whitening.T  #centered and whitened data
  # Y, W_Whitening, _, _ = whitening(X.T)
  # Y = Y.T
  Y = X

  ### calculate graph autocovariance Stilde ###
  Stilde, _ = GraphAutoCorrelation(Y, Ws)
  # Stilde = np.zeros((K, K, P))
  # for p in range(0, P): # Go over all graphs
  #   Yw = Ws[:, :, p] @ Y
  #   Yw = Yw/np.sqrt(np.mean(Yw**2, axis=0))
  #   Stilde[:, :, p] = (Y.T @ Yw)/n
  #   Stilde[:, :, p] = (Stilde[:, :, p]+Stilde[:, :, p].T)/2
  
  
  U_old = np.eye(K)
  U = np.eye(K) #scipy.stats.ortho_group.rvs(K)
  W = np.zeros((K, K))
  
  iter = 0
  while iter < maxiter:    
    YV = Y @ U.T
    W_FastICA = np.mean(G_nonlin(YV, nonlin), axis=0) * (g(YV, nonlin).T@Y)/n - np.diag(np.mean(Edgs(YV, nonlin), axis=0)).T @ U #replace with powerica
    W_GraDe = np.zeros((K, K))

    #### Graph Decorrelation Update step ### -> Here nan due to extreme values in wn
    for j in range(0, K):
      w = U[j, ].reshape(K, 1)
      wn = w.copy()
      for mi in range(0, P):
        wn = wn + 2 * (Stilde[:, :, mi] @ (w @ (w.T @ (Stilde[:, :, mi] @ w))))
      
      W_GraDe[j,] = wn[:, 0]
      if( W_FastICA[j, np.argmax(np.abs(W[j,]))] < 0):
         W_FastICA[j, ] = -W_FastICA[j, ]
      if(W_GraDe[j, np.argmax(abs(W[j, ]))] < 0):
         W_GraDe[j, ] = -W_GraDe[j, ]
  
    ### composite update step ###
    W = (1-b)*W_FastICA + b*W_GraDe
    U = np.linalg.solve(matSqrt(W@W.T), np.eye(K)).T @ W #unclear step, probably related to squared symmetric fastICA
    iter = iter+1
    if(np.linalg.norm(abs(U)-abs(U_old))<eps):
      break
    elif(iter==maxiter):
      break #stop("maxiter reached without convergence")
    U_old = U
   
  
  #GAMMA = U @ W_Whitening
  return W, Y


def matSqrt(A):
  eigval, eigvec = np.linalg.eigh(A)
  return eigvec @ (np.diag(eigval**(1/2))) @ eigvec.T


def G_nonlin(s, nonlin):  # , Huber_param, lp_param, fair_param):
    """ This function computes the ICA nonlinearity for a given input s = w.T @ x.

    Args:
        s (array): Array containing the values of (w.T @ x)
        nonlin (String): String defining the used nonlinearity

    Returns:
        G: Integral of Nonlinearity function g for used in PowerICA algorithm.
    """
    G = np.empty(np.shape(s))
    if nonlin == 'tanh':
      G = np.log(np.cosh(s))
    elif nonlin == 'pow3':
      G = 1/4 * s ** 4
    else:
      print('Invalid nonlinearity')
      pass
    return G


def g(s, nonlin): #, Huber_param, lp_param, fair_param):
    """ This function computes the ICA nonlinearity for a given input s = w.T @ x.

    Args:
        s (array): Array containing the values of (w.T @ x)
        nonlin (String): String defining the used nonlinearity

    Returns:
        g: Nonlinearity function for used in PowerICA algorithm.
    """
    g = np.empty(np.shape(s))
    if nonlin == 'tanh':
      g = np.tanh(s)
    elif nonlin == 'pow3':
      g = s ** 3
    elif nonlin == 'gaus':
      g = s * np.exp(-(1 / 2) * s ** 2)
    elif nonlin == 'skew':
      g = s ** 2
    elif nonlin == 'fair':  ## pretty shitty?! probably sth wrong maybe sweep over different A's
      fair_param = 1.3998
      g = fair_param * s / (np.absolute(s) + fair_param)
    elif nonlin == 'Pseudo-Huber':
      g = np.divide(s, np.sqrt(1 + s ** 2 / 2))
    elif nonlin == 'lp':
      lp_param = 1.5
      g = np.sign(s) * np.absolute(s) ** (lp_param - 1)
    elif nonlin == 'Huber':
      Huber_param = 1.345
      if np.linalg.norm(s) <= Huber_param:
        g = s
      elif np.linalg.norm(s) > Huber_param:
        g = Huber_param * np.sign(s)
    elif nonlin == 'bt06':
      g = np.maximum(0, s - 0.6) ** 2 - np.minimum(0, s + 0.6) ** 2
    elif nonlin == 'bt10':
      g = np.max(0, s - 1.0) ** 2 - np.min(0, s + 1.0) ** 2
    elif nonlin == 'bt12':
      g = np.max(0, s - 1.2) ** 2 - np.min(0, s + 1.2) ** 2
    elif nonlin == 'bt14':
      g = np.max(0, s - 1.4) ** 2 - np.min(0, s + 1.4) ** 2
    elif nonlin == 'bt16':
      g = np.max(0, s - 1.6) ** 2 - np.min(0, s + 1.6) ** 2
    elif nonlin == 'tan1':
      g = np.tanh(1.25 * s)
    elif nonlin == 'tan2':
      g = np.tanh(1.5 * s)
    elif nonlin == 'tan3':
      g = np.tanh(1.75 * s)
    elif nonlin == 'tan4':
      g = np.tanh(2 * s)
    elif nonlin == 'gau1':
      g = s * np.exp(-(1.07 / 2) * s ** 2)
    elif nonlin == 'gau2':
      g = s * np.exp(-(1.15 / 2) * s ** 2)
    elif nonlin == 'gau3':
      g = s * np.exp(-(0.95 / 2) * s ** 2)
    else:
      print('Invalid nonlinearity')
      pass
    return g
    ##########################################################################


def Edgs(s, nonlin): #, Huber_param, lp_param, fair_param):
  """This function computes E[g'(w^t*x)]for a given input s = w.T @ x.

  Args:
      s (array): Array containing the values of (w.T @ x)
      nonlin (String): String defining the used nonlinearity

  Returns:
      dg: Derivative of the nonlinearity used in PowerICA algorithm.
  """
  dg = np.empty(len(s))
  if nonlin == 'tanh':
    dg = 1 - np.tanh(s) ** 2
  elif nonlin == 'pow3':
    dg = 3 * s ** 2
  elif nonlin == 'gaus':
    dg = (1 - s ** 2) * np.exp(- (1 / 2) * s ** 2)
  elif nonlin == 'skew':
    dg = 0.5 * s
  elif nonlin == 'fair':
    dg = fair_param ** 2 / ((s + fair_param) ** 2)
  elif nonlin == 'Pseudo-Huber':
    dg = np.divide(1, (1 + s ** 2 / 2) ** (3 / 2))
  elif nonlin == 'lp':
    dg = (lp_param - 1) * s * np.sign(s) * np.absolute(s) ** (lp_param - 3)
  elif nonlin == 'Huber':
    if np.linalg.norm(s) <= Huber_param:
      dg = 1
    elif np.linalg.norm(s) > Huber_param:
      dg = Huber_param
  elif nonlin == 'bt06':  # pseudo Huber
    dg = 2 * np.maximum(0, s - 0.6) - 2 * np.minimum(0, s + 0.6)
  elif nonlin == 'bt10':
    dg = 2 * np.maximum(0, s - 1.0) - 2 * np.minimum(0, s + 1.0)
  elif nonlin == 'bt12':
    dg = 2 * np.maximum(0, s - 1.2) - 2 * np.minimum(0, s + 1.2)
  elif nonlin == 'bt14':
    dg = 2 * np.maximum(0, s - 1.4) - 2 * np.minimum(0, s + 1.4)
  elif nonlin == 'bt16':
    dg = 2 * np.maximum(0, s - 1.6) - 2 * np.minimum(0, s + 1.6)
  elif nonlin == 'tan1':
    dg = 1.25 * (1 - np.tanh(1.25 * s) ** 2)
  elif nonlin == 'tan2':
    dg = 1.5 * (1 - np.tanh(1.5 * s) ** 2)
  elif nonlin == 'tan3':
    dg = 1.75 * (1 - np.tanh(1.75 * s) ** 2)
  elif nonlin == 'tan4':
    dg = 2 * (1 - np.tanh(2 * s) ** 2)
  elif nonlin == 'gau1':
    dg = (1 - 1.07 * s ** 2) * np.exp(- (1.07 / 2) * s ** 2)
  elif nonlin == 'gau2':
    dg = (1 - 1.15 * s ** 2) * np.exp(- (1.15 / 2) * s ** 2)
  elif nonlin == 'gau3':
    dg = (1 - 0.95 * s ** 2) * np.exp(- (0.95 / 2) * s ** 2)
  else:
    print('Invalid nonlinearity')
    pass
  return dg




##########################
###### Small test ########
##########################
if __name__ == "__main__":

  G1 = graphs.ErdosRenyi(1000, p=0.1)
  G2 = graphs.ErdosRenyi(1000, p=0.15)
  G3 = graphs.ErdosRenyi(1000, p=0.2)
  G4 = graphs.ErdosRenyi(1000, p=0.5)
  S1 = graphs.Sensor(1000)
  S2 = graphs.Sensor(1000)
  S3 = graphs.Sensor(1000)

  Ws = np.array([G1.W.toarray(), G2.W.toarray(), G3.W.toarray(), G4.W.toarray()]).reshape(1000, 1000, 4)
  Ss = np.array([S1.W.toarray(), S2.W.toarray(), S3.W.toarray()]).reshape(1000, 1000, 3)


  data = np.stack([create_signal(x=1000, f=2, c='ecg'),
                  create_signal(x=1000, f=5, c='cos'),
                  create_signal(x=1000, f=10, c='rect'),
                  create_signal(x=1000, f=25, c='sawt')]).T

  mixmat = mixing_matrix(4)
  mixeddata = mixmat @ data.T
  white_data, W_whiten, n_comp = whitening(mixeddata, type='sample', percentile=0.999999)
  U, _ = FastICA_GraDe(white_data.T, Ws)
  print(U) # returns nan, have to check why
  unMixed_fast = U @ white_data

  i = 0
  fig1, axs1 = plt.subplots(4)
  fig2, axs2 = plt.subplots(4)

  while (i < 4):
    # input signals
    axs1[i].plot(data[:, i], lw=1)
    axs1[i].set_ylabel('sig: ' + str(i))
    fig1.suptitle('Input Signals')

    axs2[i].plot(unMixed_fast[i, :], lw=1)
    axs2[i].set_ylabel('sig: ' + str(i))
    fig2.suptitle('Unmixed Signals')

    i = i + 1

  plt.show()

