import numpy as np
import scipy
  
def powerICA(X, nonlin, seed=None,Huber_param=1.345,lp_param=1.5,fair_param=1.3998):
    """This function is implemented based on Algorithm 1 in paper below.
       S. Basiri, E. Ollila and V. Koivunen, "Alternative Derivation of 
       FastICA With Novel Power Iteration Algorithm," in IEEE Signal 
       Processing Letters, vol. 24, no. 9, pp. 1378-1382, Sept. 2017.

    Args:
        X (array):  d*n array of mixture recordings (X should be centered and whitened).
        nonlin (String): String ICA nonlinearities. The following nonlinearities are supported:
        tanh, pow3, gaus, skew, rt06, lt06, bt00, bt02, bt06, bt10, 
        bt12, bt14, bt16, tan1, tan2, tan3, tan4, gau1, gau2, gau3
        seed (integer): Set seed for reproducible initial guess. If None W0 will be different each time.

    Returns:
        W[array]: PowerICA estimate of unmixing matrix (d*d).
        flg[integer]: Returns 1 when the algorithm has successfully converged and 0 when the
        algorithm could not converge.
    """
    
    [d,n] = np.shape(X)
    I = np.eye(d)
    C = (X @ X.T)/n - I
    flg = 0
    #print(C)
    W = np.zeros((d,d), dtype=X.dtype)
    if seed is not None:
        np.random.seed(seed)
    W0 = scipy.linalg.orth(np.random.rand(d,d))     #initial guess for unmixing Matrix

    
    if d>n:
        print('Data has invalid shape! Should be d × n array with d < n!')
    if not np.isreal(X).all():
        print('Data is not real!')
    # if np.max(np.abs(C[:]))> 1e-10:
    #     print('Data is not whitened!')
    # if np.max(np.abs((np.mean(X,axis=1))))>1e-10:
    #     print('Data is not centered!')

    for k in range(0,d-1):  
        ## (1) Initialization
        w0 = W0[k,:].reshape(1,d).T
        ## (2) Compute the orthogonal operator
        Orth = (I - W.T.dot(W))
        ## (3-6) compute Node:1 and Node:2 in series
        w1, delta1, flg1 = Node1(X, nonlin, w0, Orth, Huber_param,lp_param,fair_param) 
        w2, delta2, flg2 = Node2(X, nonlin, w0, Orth, Huber_param,lp_param,fair_param)

        #print('Orth',Orth,'W',W)
        flg = flg1*flg2
        ## (7) choose the unmixing vector with larger delta
        if flg == 1:
            if np.linalg.norm(delta1) > np.linalg.norm(delta2):
                W[k,:] = w1.T
                
            else:
                W[k,:] = w2.T
        else:
            W[k:d-1,:] = np.zeros((1,d))
            flg = 0
            break
        ## (8) compute the last demixing vector
    if flg == 1:
        W[d-1,:] = (W0[d-1,:] @ (I - W.T @ W))/np.linalg.norm(W0[d-1,:] @ (I - W.T @ W))
    return W,flg


def Node1(X, nonlin, w0,Orth,Huber_param,lp_param,fair_param):
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
    s=0
    gs=0
    while i <= MaxIter:
        wOld = w
        s = w.T @ X
        gs = g(s,nonlin,Huber_param,lp_param,fair_param)
        w = (X @ gs.T)/n #(4)
        w = Orth @w #(5)
        w = w/np.linalg.norm(w); #(6)   
        if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
            #print('Node1 converged after',i,'iterations\n')
            break
        i = i + 1 #(3)
    if i <= MaxIter:
        beta = Edgs(s,nonlin,Huber_param,lp_param,fair_param)
        delta = np.absolute((s @ gs.T)/n - beta)
    else:
        print('Node1 did not converge after',MaxIter,'iterations\n')
        w = []
        flg = 0
        delta = -1
    return w, delta, flg
    ###########################################################################


def Node2(X, nonlin, w0,Orth,Huber_param,lp_param,fair_param):
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
    s_max = np.sqrt(np.sum(X**2,axis=0))
    gs_max = g(s_max,nonlin,Huber_param,lp_param,fair_param)
    c = (s_max @ gs_max.T)/n + 0.5
    while i <= MaxIter:
        wOld = w
        s = w.T @ X
        gs = g(s,nonlin,Huber_param,lp_param,fair_param)
        m = (X @ gs.T)/n
        w = m - c*w    #(4)
        w = Orth @ w     #(5)
        w = w/np.linalg.norm(w)  #(6)
        if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
            #print('Node2 converged after',i,'iterations\n')
            break
        i = i + 1  #(3)
    if i <= MaxIter:
        beta = Edgs(s,nonlin,Huber_param,lp_param,fair_param)
        delta = np.absolute((s @ gs.T)/n - beta)#gamma is delta and s@gs.T/n is gamma
    else:
        print('Node2 did not converge after',MaxIter,'iterations\n')
        w = []
        flg = 0
        delta = -1
    return w, delta, flg
    ###########################################################################


def g(s,nonlin,Huber_param,lp_param,fair_param):
    """ This function computes the ICA nonlinearity for a given input s = w.T @ x.
   

    Args:
        s (array): Array containing the values of (w.T @ x)
        nonlin (String): String defining the used nonlinearity

    Returns:
        g: Nonlinearity function for used in PowerICA algorithm.
    """
    g=np.empty(np.shape(s))
    if nonlin == 'tanh':
        g = np.tanh(s)
    elif nonlin =='pow3':
        g = s**3
    elif nonlin == 'gaus':
        g = s*np.exp(-(1/2)*s**2)
    elif nonlin =='skew':
        g = s**2
    elif nonlin =='fair':## pretty shitty?! probably sth wrong maybe sweep over different A's
        fair_param = 1.3998
        g = fair_param*s/(np.absolute(s)+fair_param)
    elif nonlin =='Pseudo-Huber':
        g = np.divide(s,np.sqrt(1+s**2/2))
    elif nonlin =='lp':
        lp_param=1.5
        g = np.sign(s)*np.absolute(s)**(lp_param-1)
    elif nonlin =='Huber':
        Huber_param = 1.345
        if np.linalg.norm(s) <= Huber_param:
            g = s
        elif np.linalg.norm(s) > Huber_param:
            g = Huber_param*np.sign(s)
    elif nonlin =='bt06':
        g = np.maximum(0,s-0.6)**2 - np.minimum(0,s+0.6)**2
    elif nonlin =='bt10':
        g = np.max(0,s-1.0)**2 - np.min(0,s+1.0)**2
    elif nonlin =='bt12':
        g = np.max(0,s-1.2)**2 - np.min(0,s+1.2)**2
    elif nonlin =='bt14':
        g = np.max(0,s-1.4)**2 - np.min(0,s+1.4)**2
    elif nonlin =='bt16':
        g = np.max(0,s-1.6)**2 - np.min(0,s+1.6)**2
    elif nonlin =='tan1':
        g = np.tanh(1.25*s)
    elif nonlin =='tan2':
        g = np.tanh(1.5*s)
    elif nonlin =='tan3':
        g = np.tanh(1.75*s)
    elif nonlin =='tan4':
        g = np.tanh(2*s)
    elif nonlin =='gau1':
        g = s*np.exp(-(1.07/2)*s**2)
    elif nonlin =='gau2':
        g = s*np.exp(-(1.15/2)*s**2)
    elif nonlin =='gau3':
        g = s*np.exp(-(0.95/2)*s**2)
    else:
        print('Invalid nonlinearity')
        pass
    return g
    ##########################################################################


def Edgs(s,nonlin,Huber_param,lp_param,fair_param):
    """This function computes E[g'(w^t*x)]for a given input s = w.T @ x.

    Args:
        s (array): Array containing the values of (w.T @ x)
        nonlin (String): String defining the used nonlinearity

    Returns:
        dg: Derivative of the nonlinearity used in PowerICA algorithm.
    """
    dg = np.empty(len(s))
    if nonlin =='tanh':
        dg = 1-np.tanh(s)**2
    elif nonlin =='pow3':
        dg = 3*s**2
    elif nonlin =='gaus':
        dg = (1 - s**2) * np.exp(- (1/2)*s**2)
    elif nonlin =='skew':
        dg =  0.5*s 
    elif nonlin =='fair':
        dg = fair_param**2/((s+fair_param)**2)
    elif nonlin =='Pseudo-Huber':
        dg = np.divide(1,(1+s**2/2)**(3/2))
    elif nonlin =='lp':
        dg = (lp_param-1)*s*np.sign(s)*np.absolute(s)**(lp_param-3)
    elif nonlin =='Huber':
        if np.linalg.norm(s) <= Huber_param:
            dg = 1
        elif np.linalg.norm(s) > Huber_param:
            dg = Huber_param
    elif nonlin =='bt06':#pseudo Huber
        dg = 2*np.maximum(0,s-0.6) - 2*np.minimum(0,s+0.6)
    elif nonlin =='bt10':
        dg =  2*np.maximum(0,s-1.0) - 2*np.minimum(0,s+1.0)
    elif nonlin =='bt12':
        dg = 2*np.maximum(0,s-1.2) - 2*np.minimum(0,s+1.2)
    elif nonlin =='bt14':
        dg = 2*np.maximum(0,s-1.4) - 2*np.minimum(0,s+1.4)
    elif nonlin =='bt16':
        dg = 2*np.maximum(0,s-1.6) - 2*np.minimum(0,s+1.6)
    elif nonlin =='tan1':
        dg = 1.25*(1-np.tanh(1.25*s)**2)
    elif nonlin =='tan2':
        dg = 1.5*(1-np.tanh(1.5*s)**2)
    elif nonlin =='tan3':
        dg = 1.75*(1-np.tanh(1.75*s)**2)
    elif nonlin =='tan4':
        dg = 2*(1-np.tanh(2*s)**2)
    elif nonlin =='gau1':
        dg = (1 - 1.07*s**2) * np.exp(- (1.07/2)*s**2)
    elif nonlin =='gau2':
        dg = (1 - 1.15*s**2) * np.exp(- (1.15/2)*s**2)
    elif nonlin =='gau3':
        dg = (1 - 0.95*s**2) * np.exp(- (0.95/2)*s**2)
    else:
        print('Invalid nonlinearity')
        pass
    return np.mean(dg)