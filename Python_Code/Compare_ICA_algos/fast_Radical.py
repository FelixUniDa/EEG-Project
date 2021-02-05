import numpy as np
from utils import *


# *****************************************************************
# Copyright (c) Erik G. Learned-Miller, 2004.
# *****************************************************************
# RADICAL   Solve the ICA problem in arbitrary dimension.
#
#    Version 1.1. Major bug fix. Faster entropy estimator.
# 
#    Apr.1, 2004. Major bug fix. Whitening matrix was wrong. Thanks
#      to Sergey Astakhov for spotting this one.
#
#    Mar.28, 2004. Speed up inner loop by about 20# with a better
#      entropy estimator.
#
#    Version 1.0. First release.
#  
#    [Yopt,Wopt] = RADICAL(X) takes a single argument, X,
#    the matrix of mixed components, with one component per
#    row, and finds the best "unmixing matrix" Wopt that it
#    can. Wopt applied to the mixed components X produces the
#    approximately independent components Yopt, with one component
#    per row.
#
#    If the input data X is 5x1000, for example, then Yopt should
#    also be 5x1000, and Wopt will be 5x5.
#    ************************************************************* 
#
#    PARAMETERS: Set these parameters by hand in the next code block.
#
#    K:        The number of angles at which to evaluate the contrast
#              function. The ICA contrast function will be evaluated
#              at K evenly spaced rotations from -Pi/4 to Pi/4. For
#              small data sets (less than a few hundred points), the
#              default value of 150 should work well. For larger data
#              sets, very small benefits may be realized by
#              increasing the value of K, but keep in mind that the
#              algorithm is linear in K.
#
#    AUG_FLAG: This flag is set to 1 by default, which indicates
#              that the data set will be "augmented" as discussed
#              in the paper. If this flag is set to 0, then the
#              data will not be augmented, and the next two 
#              arguments are meaningless. For large data
#              sets with more than 10000 points, this flag should
#              usually be set to 0, as there is usually no need to
#              augment the data in these cases.
#
#    reps:     This is the number of replicated points for each  
#              original point. The default value is 30. The larger
#              the number of points in the data set, the smaller
#              this value can be. For data sets of 10,000 points or
#              more, point replication should be de-activated by setting
#              AUG_FLAG to 0 (see above).               
#
#    stdev:    This is the standard deviation of the replicated points. I
#              can't give too much guidance as to how to set this
#              parameter, but values much larger or smaller than
#              the default don't seem to work very well in the
#              experiments I've done. 

#function [Yopt,Wopt]=RADICAL(X)
def RADICAL(X, K = 150, sweeps = 5, seed=None):

    # The recommended default parameter values are:
    # K=150
    # AUG_FLAG=1
    # reps=30
    # stdev=0.175

    # ************************************************************
    # User should change parameter values here:
    K=K
    AUG_FLAG=0
    reps=30
    stdev=0.175
    # ************************************************************

    # When AUG_FLAG is off, do not augment data. Use original data only.
    if(AUG_FLAG==0):
      reps=1


    [dim,N] = np.shape(X)
    m = np.floor(np.sqrt(N))     # m for use in m-spacing estimator.

    # ****************
    # Whiten the data. Store the whitening operation to combine with
    # rotation matrix for total solution.

    X_white = X

    sweeps = sweeps#dim-1
    oldTotalRot = np.identity(dim)
    sweepIter = 0             # Current sweep number.
    totalRot = np.identity(dim)
    xcur = X_white

    #K represents the number of rotations to examine on the FINAL sweep. To optimize performance, we
    #start with a smaller number of rotations to examine.Then, we increase the number of angles to get
    #better resolution as we get closer to the solution.For the first half of the sweeps, we use a
    #constant number for K.Then we increase it exponentially toward the finish.

    finalK = K
    startKfloat = (finalK / 1.3**(np.ceil(sweeps / 2)))
    newKfloat = startKfloat

    for sweepNum in range(0, sweeps):
        #print(1, 'Sweep # ', sweepNum,'of', sweeps)
        range1 = np.pi / 2

        # Compute number of angle samples for this sweep.

        if sweepNum > (sweeps / 2):
            newKfloat = newKfloat * 1.3
            newK = np.floor(newKfloat)
        else:
            newKfloat = startKfloat
            newK = max(30, np.floor(newKfloat))


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # Iterate over all possible Jacobi rotations.
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        for i in range(0, dim-1):
            for j in range(i+1, dim):

                #print(1, 'Unmixing dimensions', i, ' and', j)
                # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                # Extract dimensions(i, j) from the current data.
                # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                curSubSpace = np.stack((np.array(xcur[i, :]), np.array(xcur[j, :])))

                # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
                # Find the best angle theta for this Jacobi rotation.
                # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

                thetaStar, rotStar = radicalOptTheta(curSubSpace, stdev, m, reps, newK, range1)

                # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
                # Incorporate Jacobi rotation into solution.
                #** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
                newRotComponent = np.eye(dim)
                newRotComponent[i, i] = np.cos(thetaStar)
                newRotComponent[i, j] = -np.sin(thetaStar)
                newRotComponent[j, i] = np.sin(thetaStar)
                newRotComponent[j, j] = np.cos(thetaStar)
                totalRot = newRotComponent @ totalRot
                xcur = totalRot @ X_white


        oldTotalRot = totalRot

    Wopt = totalRot
    Yopt = Wopt @ X

    return Wopt


def radicalOptTheta(x,stdev,m,reps,K,range1):

    # m is the number of intervals in an m - spacing reps is the number of points used in smoothing
    # K is the number of angles theta to check for each Jacobi rotation.
    d, N = np.shape(x)

    # This routine assumes that it gets whitened data. First, we augment the points with reps near copies of each point.
    if reps == 1:
        xAug = x
    else:
        xAug = np.random.randn(d, N * reps) * stdev + np.repeat(x, reps, axis=0).reshape(d, N*reps)

    # Then we rotate this data to various angles, evaluate the sum of the marginals, and take the min.
    perc = range1 / (np.pi / 2)
    numberK = perc * K
    start = np.floor(K / 2 - numberK / 2) + 1
    endPt = int(np.ceil(K / 2 + numberK / 2))
    marginalAtTheta = np.empty((1,d))
    ent = np.empty((1, endPt))


    for i in range(0, int(K)):
        # Map theta from -pi / 4 to pi / 4 instead of 0 to pi / 2. This will allow us to use Amari - distance
        # for test of convergence.
        theta = (i - 1)/(K - 1) * np.pi/2 - np.pi/4
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotPts = rot @ xAug


        for j in range(0, d):
            marginalAtTheta[0, j] = vasicekm(rotPts[j,:], m)

        ent[0, i] = np.sum(marginalAtTheta)

    ind = np.argsort(ent)
    thetaStar = (ind[0, 1] - 1)/(K - 1) * np.pi/2 - np.pi/4
    #print(1, 'rotated', thetaStar / (2 * np.pi) * 360 ,'degrees.\n')
    rotStar = np.array([[np.cos(thetaStar), -np.sin(thetaStar)], [np.sin(thetaStar), np.cos(thetaStar)]])

    return thetaStar, rotStar


def vasicekm(v, m):
    len = np.size(v)
    vals = np.sort(v)
    m = int(m)

    #Note that the intervals overlap for this estimator.
    intvals = vals[m + 1:len]-vals[1: len - m]
    hvec = np.log(intvals)
    h = np.sum(hvec)

    return h