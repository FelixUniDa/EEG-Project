import numpy as np
import scipy.stats
  
def powerICA(X, nonlin, n_components):
    #
    # This function is implemented based on Algorithm 1 in paper below.
    #   S. Basiri, E. Ollila and V. Koivunen, "Alternative Derivation of 
    #   FastICA With Novel Power Iteration Algorithm," in IEEE Signal 
    #   Processing Letters, vol. 24, no. 9, pp. 1378-1382, Sept. 2017.
    # If you use this function in your publication please cite our paper using 
    # the above citation info.
    #
    # Matlab Code by Shahab Basiri, Aalto University 2017
    # shahab.basiri@aalto.fi
    ###########################################################################
    ## INPUT:
    #   X: d*n array of mixture recordings (X should be centered and whitened).
    #   nonlin: ICA nonlinearities. 
    #           It can be either a single string or a d*1 array of strings. 
    #           The following nonlinearities are supported.
    #           tanh, pow3, gaus, skew, rt06, lt06, bt00, bt02, bt06, bt10, 
    #           bt12, bt14, bt16, tan1, tan2, tan3, tan4, gau1, gau2, gau3.
    #   W0: the initial start of the algorithm (an orthogonal d*d matrix).
    #   mode: can be set either to 'serial' or 'parallel'. The 'serial' mode 
    #         is used when only one computing node is available or the dataset
    #         is of small size. The 'parallel' mode runs two Matlab instances
    #         on different CPU cores. It is recommended for large datasets.
    ## OUTPUT:
    #   W: PowerICA estimate of demixing matrix (d*d).
    #   flg: 1 when the algorithm has converged successfully  and 0 when the
    #   algorithm has failed to converge.
    ###########################################################################
    [d,n] = np.shape(X)
    I = np.identity(d)
    C = X@X.T/n - I
    print(C)
    W = np.zeros((d,d), dtype=X.dtype)
    W0 = scipy.stats.ortho_group.rvs(n_components)
    ### Default settings
    #if d>n:
    #     print('Data should be a d × n array with d < n!')
    #if not np.isreal(X).all():
    #     print('Data is not real!')
    #if np.max(np.abs(C[:]))>10:#1e-10:
    #     print('Data should be whitened!')
    # if max(abs((np.mean(X,2))))>1:#e-10:
    #      print('Data should be centered!')
    ###########################################################################
    ## Serial mode
    for k in range(1,d):  
        ## (1) initialize 
        w0 = W0[k,:].T
        ## (2) compute the orthogonal operator
        Orth = (I - W.T@W) 
        ## (3-6) compute Node:1 and Node:2 in series
        w1, gamma1, flg1 = Node1(X, nonlin, w0, Orth) 
        w2, gamma2, flg2 = Node2(X, nonlin, w0, Orth)
        flg = flg1*flg2
        ## (7) choose the demixing vector with larger gamma
        if flg == 1:
            if np.linalg.norm(gamma1) > np.linalg.norm(gamma2):
                W[k,:] = w1.T
            else:
                W[k,:] = w2.T
        else:
            W[k:d-1,:] = []
            flg = 0
            break
        ## (8) compute the last demixing vector
        print(W0,I,W.T)
        if flg == 1:
             W[d-1,:] = W0[d-1,:] @ (I - W.T@W)/np.linalg.norm(W0[d-1,:] @ (I - (W.T @ W)))
    return  W
    ###########################################################################    
   
    ###########################################################################
    #                       Auxiliary functions
    ###########################################################################
def Node1(X, nonlin, w0,Orth):
    a, n = np.shape(X)
    ################################
    MaxIter = 10000
    epsilon = 0.0001
    flg = 1
    i = 1
    w = w0
    s=0
    gs=0
    while i <= MaxIter:
        wOld = w
        s = np.dot(w.T,X)
        gs = g(s,nonlin)
        w = np.dot(X,gs.T)/n #(4)
        w = np.dot(Orth,w) #(5)
        w = w/np.linalg.norm(w); #(6)   
        if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
            break
        i = i + 1 #(3)
    #     fprintf('IC converged after #d iterations\n',i);
    if i <= MaxIter:
        beta = Edgs(s,nonlin)
        gamma = np.absolute(np.dot(s,gs.T)/n - beta)
    else:
        #     fprintf('IC did not converged after #d iterations\n',MaxIter);
        w = []
        flg = 0
        gamma = -1
    return w, gamma, flg
    ###########################################################################
def Node2(X, nonlin, w0,Orth):
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
    s_max = np.sqrt(np.sum(X**2))
    gs_max = g(s_max,nonlin)
    c = (s_max*gs_max.T)/n + 0.5
    while i <= MaxIter:
        wOld = w
        s = w.T@X
        gs = g(s,nonlin)
        m = X@gs.T/n
        w = m - np.dot(c,w)    #(4)
        w = Orth@w     #(5)
        w = w/np.linalg.norm(w)  #(6)
        if np.linalg.norm(w - wOld) < epsilon or np.linalg.norm(w + wOld) < epsilon:
            break
        i = i + 1  #(3)
        #print('IC converged after #d iterations\n',i)
    if i <= MaxIter:
        beta = Edgs(s,nonlin)
        gamma = np.absolute(s@gs.T/n - beta)
    else:
        #print('IC did not converged after #d iterations\n',MaxIter)
        w = []
        flg = 0
        gamma = -1
    return w, gamma, flg
    ###########################################################################
def g(s,nonlin):
    g=0
    # This function computes the ICA nonlinearity for a given input s = w^t*x.
    if nonlin == 'tanh':
        g = np.tanh(s)
    elif nonlin =='pow3':
        g = s**3
    elif nonlin == 'gaus':
        g = s*np.exp(-(1/2)*s**2)
    elif nonlin =='skew':
        g = s**2
    elif nonlin =='rt06':
        g = np.max(0,s-0.6)**2
    elif nonlin =='lt06':
        g = np.min(0,s+0.6)**2
    elif nonlin =='bt00':
        g = np.max(0,s)**2 - np.min(0,s)**2
    elif nonlin =='bt02':
        g = np.max(0,s-0.2)**2 - np.min(0,s+0.2)**2
    elif nonlin =='bt06':
        g = np.max(0,s-0.6)**2 - np.min(0,s+0.6)**2
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
def Edgs(s,nonlin):
    # This function computes E[g'(w^t*x)]for a given input s = w^t*x.
    dg = 0
    if nonlin =='tanh':
        dg = 1-np.tanh(s)**2
    elif nonlin =='pow3':
        dg = 3*s**2
    elif nonlin =='gaus':
        dg = (1 - s**2) * np.exp(- (1/2)*s**2)
    elif nonlin =='skew':
        dg =  0 
    elif nonlin =='rt06':
        dg =  2*np.max(0,s-0.6)
    elif nonlin =='lt06':
        dg = 2*np.min(0,s+0.6)
    elif nonlin =='bt00':
        dg = 2*np.max(0,s) - 2*np.min(0,s)
    elif nonlin =='bt02':
        dg = 2*np.max(0,s-0.2) - 2*np.min(0,s+0.2)
    elif nonlin =='bt06':
        dg = 2*np.max(0,s-0.6) - 2*np.min(0,s+0.6)
    elif nonlin =='bt10':
        dg =  2*np.max(0,s-1.0) - 2*np.min(0,s+1.0)
    elif nonlin =='bt12':
        dg = 2*np.max(0,s-1.2) - 2*np.min(0,s+1.2)
    elif nonlin =='bt14':
        dg = 2*np.max(0,s-1.4) - 2*np.min(0,s+1.4)
    elif nonlin =='bt16':
        dg = 2*np.max(0,s-1.6) - 2*np.min(0,s+1.6)
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

    return np.mean(dg)