import numpy as np
from gurobipy import *

def damp(X, R):
    # X should be n-by-m, n=dimension, m=samplesize
    Z = np.random.random(X.shape[1])
    p = np.true_divide(-1*np.sum(np.power(X,2),0), np.power(R,2))
    threshold = np.exp(p.getA()[0])

    Xdamp = X[:,Z <= threshold]
    Xrejected = X[:, Z > threshold]
    rate = np.true_divide(Xdamp.shape[1], X.shape[1])

    return (Xdamp, Xrejected, rate)

def centroidOrthogonalizer(X):
    [n,m] = X.shape
    minkowski = np.zeros(m)

    model = Model("centroid")
    lambdas = [None] * m
    
    lam = model.addVar(name="lambda")

    for i in range(len(lambdas)):
        lambdas[i] = model.addVar(name="lambda_"+str(i))

    model.update()

    for i in range(len(lambdas)):
        model.addConstr(lambdas[i] <= 1, "lam "+str(i)+" <= 1")
        model.addConstr(lambdas[i]  >= -1, "lam "+str(i)+" >= -1")

    model.addConstr(lam >= 0)

    model.setObjective(lam, GRB.MAXIMIZE)

    N = np.true_divide(1, m)
    pointConstraints = [None]*n

    for i in range(m):
        print("Starting new point query...")
        model.reset()

        for j in range(n):
            print("Adding dimension "+str(j)+" constraint")
            if pointConstraints[j] != None:
                model.remove(pointConstraints[j])
            # TODO: try using LinExpr instead of np.dot
            pointConstraints[j] = model.addConstr(N*np.dot(X[j,:], lambdas) == lam*X[j, i], "cz"+str(j))

        print("Optimizing...")
        model.optimize()

        minkowski[i] = np.true_divide(1, model.objVal)

    thresh = np.percentile(minkowski)
    filteredX = X[:, minkowski <= thresh]

    C = np.true_divide((np.mat(filteredX) * np.mat(filteredX).T), m)
    return np.linalg.inv(scipy.linalg.sqrtm(C))
