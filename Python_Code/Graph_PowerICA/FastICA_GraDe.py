#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pygsp import graphs
from utils import *

def FastICA_GraDe(X, Ws, G = None, g = None, dg = None, b=1, eps=1e-06, maxiter=100):
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
  n = np.shape(X)[0]
  K = np.shape(X)[1]
  P = np.shape(Ws)[2]
  

  ###           prewhitening            ###
  ### We can use our whitening function ###
  mu = np.mean(X, axis=0)
  cov = np.cov(X, rowvar=False)
  d,E = np.linalg.eig(cov)
  W_Whitening = E @ np.diag(d**(-0.5))@ E.T
  X_centered = X - mu
  Y = X_centered @ W_Whitening.T

  ### calculate graph autocovariance Stilde ###
  Stilde = np.zeros((K,K,P))
  for p in range(0,P):
    Yw = Ws[:,:,p] @ Y  
    Yw = Yw/np.sqrt(np.mean(Yw**2))
    Stilde[:,:,p] = (Y.T @ Yw)/n 
    Stilde[:,:,p] = (Stilde[:,:,p]+Stilde[:,:,p].T)/2
  
  
  U_old = np.eye(K)
  U = scipy.stats.ortho_group.rvs(K)
  W = np.zeros((K,K))
  
  iter = 0
  while iter < maxiter:    
    # YV = Y @ U.T
    # W_FastICA = np.mean(G(YV),axis=1)*(g(YV).T@Y)/n-np.diag(np.mean(dg(YV))).T @ U #replace with powerica
    W_FastICA = np.zeros((K,K))
    W_GraDe = np.zeros((K,K))

    #### Graph Decorrelation Update step ###
    for j in range(0,K):
      w = U[j,].reshape(K,1)
      wn = w
      for p in range(0,P):
        wn = wn + 2 * (Stilde[:,:,p] @ (w @ (w.T @ (Stilde[:,:,p] @ w))))
      
      W_GraDe[j,] = wn[:,0]
      if( W_FastICA[j,np.where(np.max(np.abs(W[j,])))] < 0):
         W_FastICA[j,] = -W_FastICA[j,]
      if(W_GraDe[j,np.where(np.max(abs(W[j,])))]<0):
         W_GraDe[j,] = -W_GraDe[j,]
  
    ### composite update step ###
    W = (1-b)*W_FastICA+(b)*W_GraDe
    
    U = W# np.linalg.inv(np.sqrt(W@W.T)).T @ W #unclear step, probably related to squared symmetric fastICA
    iter = iter+1
    if(np.linalg.norm(abs(U)-abs(U_old))<eps):
      break
    elif(iter==maxiter):
      break #stop("maxiter reached without convergence")
    U_old = U
   
  
  GAMMA = U @ W_Whitening
  return GAMMA




##########################
###### Small test ########
##########################
if __name__ == "__main__":

  G1 = graphs.ErdosRenyi(1000, p=0.1)
  G2 = graphs.ErdosRenyi(1000, p=0.15)
  G3 = graphs.ErdosRenyi(1000, p=0.2)

  Ws = np.array([G1.W.toarray(),G2.W.toarray(),G3.W.toarray()]).reshape(1000,1000,3)


  data = np.stack([create_signal(f=2,c='ecg'),
                  create_signal(f=5,c='cos'),
                  create_signal(f=10,c='rect'), 
                  create_signal(f=25,c='sawt')]).T

  U = FastICA_GraDe(data,Ws)
  print(U) # returns nan, have to check why


