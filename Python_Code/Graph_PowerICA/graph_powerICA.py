def FastICA_GraDe(X, Ws, G, g, dg, b, eps=1e-06, maxiter=1000):

    n = nrow(X)
    p = ncol(X)
    K = dim(Ws)[3]

    MEAN =colMeans(X)
    COV =cov(X)
    EVD =eigen(COV, symmetric=TRUE)
    COV.sqrt.i =EVD$vectors % * % tcrossprod(diag(EVD$values ^ (-0.5)), EVD$vectors)
    X.C =sweep(X, 2, MEAN, "-")
    Y =tcrossprod(X.C, COV.sqrt.i)
    R =array(0, c(p, p, K))
    for (k in 1:K):
        Yw =Ws[,, k] % * % Y
        Yw =Yw / sqrt(mean(Yw ^ 2))
        R[, , k] =crossprod(Y, Yw) / n
        R[, , k] =(R[, , k]+t(R[, , k])) / 2

    V0 =diag(p)
    V =diag(p)
    W =matrix(0, p, p)

    iter =0
    while (TRUE)
        iter = iter+1

        YV =tcrossprod(Y, V)
        W1 =colMeans(G(YV)) * (crossprod(g(YV), Y) / n-crossprod(diag(colMeans(dg(YV))), V))

        W2 =matrix(0, p, p)
        for (j in 1:p):
            wn =w =V[j,]
            for (mi in 1:K):
                wn =wn + 2 * R[,, mi] % * % w % * % t(w) % * % R[,, mi] % * % w

            W2[j,] =wn
            if (W1[j, which.max(abs(W[j,]))] < 0):
                 W1[j,] =-W1[j,]
            if (W2[j, which.max(abs(W[j,]))] < 0):
                W2[j,] =-W2[j,]

        W =b * W1 + (1 - b) * W2

        V =crossprod(solve(mat.sqrt(tcrossprod(W, W))), W)

        if (mat.norm(abs(V) - abs(V0)) < eps):
            break
        if (iter == maxiter):
            break  # stop("maxiter reached without convergence")
        V0 =V

    list(W=crossprod(t(V), COV.sqrt.i))