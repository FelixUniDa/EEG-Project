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

from utils import *

def FastICA_GraDe(X, Ws, G = None, g = None, dg = None, b=0.98, eps=1e-06, maxiter=1000):
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
  Y, W_Whitening, _, _ = whitening(X.T)
  Y = Y.T

  ### calculate graph autocovariance Stilde ###
  Stilde = np.zeros((K, K, P))
  for p in range(0, P): # Go over all graphs
    #Ws1 = Ws[:, :, p]
    Yw = Ws[:, :, p] @ Y
    Yw = Yw/np.sqrt(np.mean(Yw**2, axis=0))
    Stilde[:, :, p] = (Y.T @ Yw)/n
    Stilde[:, :, p] = (Stilde[:, :, p]+Stilde[:, :, p].T)/2
  
  
  U_old = np.eye(K)
  U = scipy.stats.ortho_group.rvs(K)
  W = np.zeros((K, K))
  
  iter = 0
  while iter < maxiter:    
    #YV = Y @ U.T
    #W_FastICA = np.mean(G(YV), axis=1)*(g(YV).T@Y)/n-np.diag(np.mean(dg(YV))).T @ U #replace with powerica
    W_FastICA = np.zeros((K, K))
    W_GraDe = np.zeros((K, K))

    #### Graph Decorrelation Update step ### -> Here nan due to extreme values in wn
    for j in range(0, K):
      w = U[j, ].reshape(K, 1)
      wn = w
      for p in range(0, P):
        wn = wn + 2 * (Stilde[:, :, p] @ (w @ (w.T @ (Stilde[:, :, p] @ w))))
      
      W_GraDe[j,] = wn[:, 0]
      if( W_FastICA[j, np.where(np.max(np.abs(W[j,])))] < 0):
         W_FastICA[j, ] = -W_FastICA[j,]
      if(W_GraDe[j, np.where(np.max(abs(W[j,])))] < 0):
         W_GraDe[j, ] = -W_GraDe[j, ]
  
    ### composite update step ###
    W = (1-b)*W_FastICA + (b)*W_GraDe
    
    U = W# np.linalg.inv(np.sqrt(W@W.T)).T @ W #unclear step, probably related to squared symmetric fastICA
    iter = iter+1
    if(np.linalg.norm(abs(U)-abs(U_old))<eps):
      break
    elif(iter==maxiter):
      break #stop("maxiter reached without convergence")
    U_old = U
   
  
  GAMMA = U @ W_Whitening
  return GAMMA, Y




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

  U, white_data = FastICA_GraDe(data, Ws)
  print(U) # returns nan, have to check why
  unMixed_fast = U @ white_data.T

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









def Node1(X, nonlin, w0, Orth):
    """Computes largest value of non-Gaussianity measure delta = |gamma-beta| for
       a rowvector of the estimated unmixing matrix that is a local maximizer
       of the eigenvalue gamma with largest Euclidean distance to to the bulk of
       other eigenvalues.

    Args:
        X (array): Array of the centerd and whitened mixed data with shape d x n  where d < n.
        nonlin (String): String defining the used nonlinearity.
        w0 (array): Initial guess for the rowvector w of the unmixing matrix M.T
        Orth (array): Orthogonal operator that projects onto the
        orthogonal complement of the subspace

    Returns:
        w[array]: Estimated rowvector after algorithm converges.
        delta[float]: Non-Gaussianity measure for local maximizer of gamma.
        flg[integer]: Returns 1 when the algorithm has successfully converged and 0 when the
        algorithm could not converge.

    """
    a, n = np.shape(X)
    ################################
    MaxIter = 100000
    epsilon = 0.0001
    flg = 1
    i = 1
    w = w0
    s = 0
    gs = 0
    while i <= MaxIter:
      wOld = w
      s = w.T @ X
      gs = g(s, nonlin)
      w = (X @ gs.T) / n  # (4)
      w = Orth @ w  # (5)
      w = w / np.linalg.norm(w);  # (6)
      if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
        # print('Node1 converged after',i,'iterations\n')
        break
      i = i + 1  # (3)
    if i <= MaxIter:
      beta = Edgs(s, nonlin)
      delta = np.absolute((s @ gs.T) / n - beta)
    else:
      print('Node1 did not converge after', MaxIter, 'iterations\n')
      w = []
      flg = 0
      delta = -1
    return w, delta, flg
    ###########################################################################


def Node2(X, nonlin, w0, Orth):
  """Computes largest value of non-Gaussianity measure delta = |gamma-beta| for
     a rowvector of the estimated unmixing matrix that is a local minimizer
     of the eigenvalue gamma with largest Euclidean distance to to the bulk of
     other eigenvalues.

  Args:
      X (array): Array of the centerd and whitened mixed data with shape d x n  where d < n.
      nonlin (String): String defining the used nonlinearity.
      w0 (array): Initial guess for the rowvector w of the unmixing matrix M.T.
      Orth (array): Orthogonal operator that projects onto the
      orthogonal complement of the subspace.

  Returns:
      w[array]: Estimated rowvector after algorithm converges
      delta[float]: Non-Gaussianity measure for local minimizer of gamma
      flg[integer]: Returns 1 when the algorithm has successfully converged and 0 when the
      algorithm could not converge.

  """
  a, n = np.shape(X)
  ################################
  MaxIter = 10000
  epsilon = 0.0001
  flg = 1
  i = 1
  w = w0
  s = 0
  gs = 0
  # Compute the upper bound
  s_max = np.sqrt(np.sum(X ** 2, axis=0))
  gs_max = g(s_max, nonlin)
  c = (s_max @ gs_max.T) / n + 0.5
  while i <= MaxIter:
    wOld = w
    s = w.T @ X
    gs = g(s, nonlin)
    m = (X @ gs.T) / n
    w = m - c * w  # (4)
    w = Orth @ w  # (5)
    w = w / np.linalg.norm(w)  # (6)
    if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
      # print('Node2 converged after',i,'iterations\n')
      break
    i = i + 1  # (3)
  if i <= MaxIter:
    beta = Edgs(s, nonlin)
    delta = np.absolute((s @ gs.T) / n - beta)  # gamma is delta and s@gs.T/n is gamma
  else:
    print('Node2 did not converge after', MaxIter, 'iterations\n')
    w = []
    flg = 0
    delta = -1
  return w, delta, flg

  ###########################################################################



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
  return np.mean(dg)


