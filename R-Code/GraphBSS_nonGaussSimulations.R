install.packages(c("JADE","fICA"))
library(JADE)
library(fICA)

genrAs <- function(n, mc=0.01)
{
  Il <- matrix(1, n, n)
  Il[upper.tri(Il,diag=TRUE)] <- 0
  
  lt <- which(Il>0.5)
  
  ind <- rbinom(n*(n-1)/2,1,mc)
  
  A <- matrix(0, n, n) # weight/shift matrix
  A[lt[which(ind!=0)]] <- 1
  A+t(A)
}


GenWs <- function(A, e1, e2, w)
{
  n <- nrow(A)
  A[upper.tri(A,diag=TRUE)] <- 0
  Il <- matrix(1, n, n)
  Il[upper.tri(Il,diag=TRUE)] <- 0
  
  nz <- which(abs(A)>0.0001)
  l <- length(nz)
  lt <- which(Il>0.5)
  z <- setdiff(lt,nz)
  
  W <- w*(abs(A)>0.0001)
  
  ind1 <- rbinom(length(nz),1,e1)
  ind2 <- rbinom(length(z),1,e2)
  
  W[nz[which(ind1!=0)]] <- 0  
  
  W[z[which(ind2!=0)]] <- w
  
  W+t(W)
}


GraDeDiffW <- function(X, K, Ws, eps=1e-6, maxiter=1000)
{
  n <- nrow(X)
  p <- ncol(X)
  M <- dim(Ws)[3]
  sumD <- 0
  
  MEAN <- colMeans(X)
  COV <- cov(X)
  EVD <- eigen(COV, symmetric = TRUE)
  COV.sqrt.i <- EVD$vectors %*% 
    tcrossprod(diag(EVD$values^(-0.5)), EVD$vectors)
  X.C <- sweep(X, 2, MEAN, "-")
  Y <- tcrossprod(X.C, COV.sqrt.i)
  Wsk <- array(0, c(n,n,M,K))
  R <- array(0, c(p,p,M*K))
  for(m in 1:M){
    Yw <- Ws[,,m]%*%Y 
    Wsk[,,m,1] <- Ws[,,m]
    Yw <- sweep(Yw, 2, MEAN, "-")
    Yw <- Yw/sqrt(mean(Yw^2))
    R[,,(m-1)*K+1] <- crossprod(Y,Yw)/n 
    R[,,(m-1)*K+1] <- (R[,,(m-1)*K+1]+t(R[,,(m-1)*K+1]))/2
  
    if(K>1){
      for(k in 2:K){
        Wsk[,,m,k] <- Ws[,,m]%*%Wsk[,,m,k-1]
        Yw <- Wsk[,,m,k]%*%Y  
        Yw <- Yw/sqrt(mean(diag(cov(Yw)))) 
        R[,,(m-1)*K+k] <- crossprod(Y,Yw)/n 
        R[,,(m-1)*K+k] <- (R[,,(m-1)*K+k]+t(R[,,(m-1)*K+k]))/2
      }
    }
  }
  
  res <- rjd(R, eps, maxiter)
  
  for(k in 1:K){
    sumD <- sumD+sum(diag(res$D[,,k]^2))
  }
  
  U <- res$V
  list(V=crossprod(U, COV.sqrt.i), sumD=sumD) 
}

mat.sqrt <- function(A)
{
  eig <- eigen(A, symmetric=TRUE)
  eig$vectors%*%(diag(eig$values^(1/2)))%*%t(eig$vectors)
}

mat.norm <- function(A)
{
  sqrt(sum(A^2))
}

FastICA_GraDe <- function(X, Ws, G, g, dg, b, eps=1e-06, maxiter=1000)
{
  n <- nrow(X)
  p <- ncol(X)
  K <- dim(Ws)[3]
  
  MEAN <- colMeans(X)
  COV <- cov(X)
  EVD <- eigen(COV, symmetric = TRUE)
  COV.sqrt.i <- EVD$vectors %*% 
    tcrossprod(diag(EVD$values^(-0.5)), EVD$vectors)
  X.C <- sweep(X, 2, MEAN, "-")
  Y <- tcrossprod(X.C, COV.sqrt.i)
  R <- array(0, c(p,p,K))
  for(k in 1:K){
    Yw <- Ws[,,k]%*%Y  
    Yw <- Yw/sqrt(mean(Yw^2))
    R[,,k] <- crossprod(Y,Yw)/n 
    R[,,k] <- (R[,,k]+t(R[,,k]))/2
  }
   
  V0 <- diag(p)
  V <- diag(p)
  W <- matrix(0,p,p)
  
  iter <- 0
  while (TRUE){
    iter <- iter+1
    
    YV <- tcrossprod(Y,V)
    W1 <- colMeans(G(YV))*(crossprod(g(YV),Y)/n-crossprod(diag(colMeans(dg(YV))),V))
    
    W2 <- matrix(0,p,p)
    for(j in 1:p){
      wn <- w <- V[j,]
      for(mi in 1:K){
        wn <- wn+2*R[,,mi]%*%w%*%t(w)%*%R[,,mi]%*%w
      }
      W2[j,] <- wn
      if(W1[j,which.max(abs(W[j,]))]<0) W1[j,] <- -W1[j,]
      if(W2[j,which.max(abs(W[j,]))]<0) W2[j,] <- -W2[j,]
    }
    
    W <- b*W1+(1-b)*W2
    
    V <- crossprod(solve(mat.sqrt(tcrossprod(W,W))),W)
    
    if(mat.norm(abs(V)-abs(V0))<eps) break
    if(iter==maxiter) break #stop("maxiter reached without convergence")
    V0 <- V
  } 
  
  list(W=crossprod(t(V), COV.sqrt.i)) 
} 


JADE_GraDe <- function(X, Ws, b, eps=1e-6, maxiter=1000)
{
  n <- nrow(X)
  p <- ncol(X)
  K <- dim(Ws)[3]
  sumD <- 0
  
  MEAN <- colMeans(X)
  COV <- cov(X)
  EVD <- eigen(COV, symmetric = TRUE)
  COV.sqrt.i <- EVD$vectors %*% 
    tcrossprod(diag(EVD$values^(-0.5)), EVD$vectors)
  X.C <- sweep(X, 2, MEAN, "-")
  Y <- tcrossprod(X.C, COV.sqrt.i)
  R <- array(0, c(p,p,K+p*(p+1)/2))
  for(k in 1:K){
    Yw <- Ws[,,k]%*%Y  
    #  Yw <- Yw/sqrt(mean(diag(cov(Yw)))) 
    Yw <- sweep(Yw, 2, MEAN, "-")
    Yw <- Yw/sqrt(mean(Yw^2))
    R[,,k] <- crossprod(Y,Yw)/n 
    R[,,k] <- sqrt(1-b)*(R[,,k]+t(R[,,k]))/2
  }
  
  Ip <- diag(p)
  Qij <- matrix(0, p, p)
  Yi <- numeric(p)
  Yj <- numeric(p)
  scale2 <- rep(1,p)/n
  
  l <- 1
  for (i in 1:p){
    Yi <- Y[,i]
    Qij <- crossprod((tcrossprod((Yi *Yi), scale2) * Y), Y) - Ip-2* tcrossprod(Ip[,i])
    R[,,K+l] <- sqrt(b)*Qij
    l <- l+1
    if (i>1){ 
      for (j in (1:(i-1))){
        Yj <- Y[,j]
        Qij <- crossprod((tcrossprod((Yi *Yj), scale2) * Y), Y) - tcrossprod(Ip[,i], Ip[,j])- tcrossprod(Ip[,j], Ip[,i])
        R[,,K+l] <- sqrt(b)*sqrt(2)*Qij
        l <- l+1
      }
    }
  }
   
  U <- rjd(R, eps, maxiter)$V
  
  list(V=crossprod(U, COV.sqrt.i)) 
}




##########################################################################################

repet <- 1000
MDs1 <- array(0,c(repet,3,6))
Ns <- c(250,500,1000)
P <- 4
m <- 0.05 
set.seed(932)
for(l in 1:3){
 N <- Ns[l]
 for(i in 1:repet){
  A1 <- genrAs(N, mc=m)
  # W1 <- GenWs(A1,0.8,0.2,1)
  A2 <- A1%*%A1 
  Ws <- array(0,c(N,N,8))
  Ws[,,1] <- A1
  Ws[,,2] <- A1
  Ws[,,3] <- A1
  Ws[,,4] <- A1
  Ws[,,5] <- A2
  Ws[,,6] <- A2
  Ws[,,7] <- A2
  Ws[,,8] <- A2
  
  e <- matrix(0,N,P)
  e[,1] <- rt(N,5)
  e[,2] <- rt(N,10)
  e[,3] <- rt(N,15)
  e[,4] <- rnorm(N)
  
  theta <- c(0.02,0.04,0.06,0.08)
  S <- matrix(0, N, P)
  for(j in 1:P){
    S[,j] <- e[,j]+theta[j]*Ws[,,j]%*%e[,j]
  } 
  B <- matrix(rnorm(P^2),P,P)
  X <- S%*%t(B)
 
  W1 <- tryCatch(fICA(X,method = "sym",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  W2 <- tryCatch(JADE(X,P)$W, error=function(e) matrix(1,P,P))
  W3 <- tryCatch(GraDeDiffW(X,1,Ws)$V, error=function(e) matrix(1,P,P))
  W4 <- tryCatch(FastICA_GraDe(X,Ws,Gf[[2]],gf[[2]],dgf[[2]],b=0.998)$W, error=function(e) matrix(1,P,P))
  W5 <- tryCatch(JADE_GraDe(X,Ws,b=0.2)$V, error=function(e) matrix(1,P,P))
  W6 <- tryCatch(fICA(X,method = "sym2",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  
  MDs1[i,l,1] <- MD(W1,B)
  MDs1[i,l,2] <- MD(W2,B)
  MDs1[i,l,3] <- MD(W3,B)
  MDs1[i,l,4] <- MD(W4,B)
  MDs1[i,l,5] <- MD(W5,B)
  MDs1[i,l,6] <- MD(W6,B)
  if((i/100)==floor(i/100)) print(paste("i=",i))
 }
}

#write(MDs1,file="model1_data.dat")

##########################################################################################

repet <- 1000
MDs2 <- array(0,c(repet,3,6))
Ns <- c(250,500,1000)
P <- 4
m <- 0.05
set.seed(2137)
for(l in 1:3){
 for(i in 1:repet){
  N <- Ns[l]
  A1 <- genrAs(N, mc=m)
  A2 <- A1%*%A1 
  Ws <- array(0,c(N,N,8))
  Ws[,,1] <- A1
  Ws[,,2] <- A1
  Ws[,,3] <- A1
  Ws[,,4] <- A1
  Ws[,,5] <- A2
  Ws[,,6] <- A2
  Ws[,,7] <- A2
  Ws[,,8] <- A2
  
  e <- matrix(0,N,P)
  e[,1] <- rt(N,5)
  e[,2] <- runif(N,-1,1)
  e[,3] <- rexp(N)
  e[,4] <- rnorm(N)
  
  theta <- c(0.05,0.06,0.07,0.08)
  S <- matrix(0, N, P)
  for(j in 1:P){
    S[,j] <- e[,j]+theta[j]*Ws[,,j]%*%e[,j]
  } 
  B <- matrix(rnorm(P^2),P,P)
  X <- S%*%t(B)
  
  W1 <- tryCatch(fICA(X,method = "sym",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  W2 <- tryCatch(JADE(X,P)$W, error=function(e) matrix(1,P,P))
  W3 <- tryCatch(GraDeDiffW(X,1,Ws)$V, error=function(e) matrix(1,P,P))
  W4 <- tryCatch(FastICA_GraDe(X,Ws,Gf[[2]],gf[[2]],dgf[[2]],b=0.998)$W, error=function(e) matrix(1,P,P))
  W5 <- tryCatch(JADE_GraDe(X,Ws,b=0.2)$V, error=function(e) matrix(1,P,P))
  W6 <- tryCatch(fICA(X,method = "sym2",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  
  MDs2[i,l,1] <- MD(W1,B)
  MDs2[i,l,2] <- MD(W2,B)
  MDs2[i,l,3] <- MD(W3,B)
  MDs2[i,l,4] <- MD(W4,B)
  MDs2[i,l,5] <- MD(W5,B)
  MDs2[i,l,6] <- MD(W6,B)
  
  if((i/100)==floor(i/100)) print(paste("i=",i))
 }
} 

write(MDs2,file="model2_data.dat")

##########################################################################################

repet <- 1000
MDs3 <- array(0,c(repet,3,6))
Ns <- c(250,500,1000)
P <- 4
m <- 0.05

set.seed(6815)
for(l in 1:3){
 N <- Ns[l]

 for(i in 1:repet){
  A1 <- genrAs(N, mc=m)
   
  Ws <- array(0,c(N,N,8))
  Ws[,,1] <- GenWs(A1,0.8,0.2,1)
  Ws[,,2] <- GenWs(A1,0.8,0.2,1)
  Ws[,,3] <- GenWs(A1,0.8,0.2,1)
  Ws[,,4] <- A1
  Ws[,,5] <- Ws[,,1]%*%Ws[,,1]
  Ws[,,6] <- Ws[,,2]%*%Ws[,,2]
  Ws[,,7] <- Ws[,,3]%*%Ws[,,3]
  Ws[,,8] <- A1%*%A1
  e <- matrix(0,N,P)
  e[,1] <- rt(N,15)
  e[,2] <- rt(N,15)
  e[,3] <- rt(N,15)
  e[,4] <- rt(N,15)
  
  theta <- c(0.05,0.05,0.05,0.05)
  S <- matrix(0, N, P)
  for(j in 1:P){
    S[,j] <- e[,j]+theta[j]*Ws[,,j]%*%e[,j]
  } 
  B <- matrix(rnorm(P^2),P,P)
  X <- S%*%t(B)
  
  W1 <- tryCatch(fICA(X,method = "sym",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  W2 <- tryCatch(JADE(X,P)$W, error=function(e) matrix(1,P,P))
  W3 <- tryCatch(GraDeDiffW(X,1,Ws)$V, error=function(e) matrix(1,P,P))
  W4 <- tryCatch(FastICA_GraDe(X,Ws,Gf[[2]],gf[[2]],dgf[[2]],b=0.998)$W, error=function(e) matrix(1,P,P))
  W5 <- tryCatch(JADE_GraDe(X,Ws,b=0.2)$V, error=function(e) matrix(1,P,P))
  W6 <- tryCatch(fICA(X,method = "sym2",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
  
  MDs3[i,l,1] <- MD(W1,B)
  MDs3[i,l,2] <- MD(W2,B)
  MDs3[i,l,3] <- MD(W3,B)
  MDs3[i,l,4] <- MD(W4,B)
  MDs3[i,l,5] <- MD(W5,B)
  MDs3[i,l,6] <- MD(W6,B)
  if((i/100)==floor(i/100)) print(paste("i=",i))
 }
}

write(MDs3,file="model3_data.dat")

##########################################################


repet <- 1000
MDs4 <- array(0,c(repet,3,6))
Ns <- c(250,500,1000)
P <- 4
m <- 0.05 
set.seed(1512)
for(l in 1:3){
  N <- Ns[l]
  for(i in 1:repet){
    A1 <- genrAs(N, mc=m)
    # W1 <- GenWs(A1,0.8,0.2,1)
    A2 <- A1%*%A1
      
    Ws <- array(0,c(N,N,8))
    Ws[,,1] <- A1
    Ws[,,2] <- A1
    Ws[,,3] <- A1
    Ws[,,4] <- A1
    Ws[,,5] <- A2
    Ws[,,6] <- A2
    Ws[,,7] <- A2
    Ws[,,8] <- A2
    
    e <- matrix(0,N,P)
    e[,1] <- rt(N,5)
    e[,2] <- rnorm(N)
    e[,3] <- runif(N,-1,1)
    e[,4] <- rnorm(N)
    
    theta <- c(0.04,0.04,0.08,0.08)
    S <- matrix(0, N, P)
    for(j in 1:P){
      S[,j] <- e[,j]+theta[j]*Ws[,,j]%*%e[,j]
      #S[,j] <- e[,j]+theta[j]*A1%*%e[,j]
    } 
    B <- matrix(rnorm(P^2),P,P)
    X <- S%*%t(B)
    
    W1 <- tryCatch(fICA(X,method = "sym",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
    W2 <- tryCatch(JADE(X,P)$W, error=function(e) matrix(1,P,P))
    W3 <- tryCatch(GraDeDiffW(X,1,Ws)$V, error=function(e) matrix(1,P,P))
    W4 <- tryCatch(FastICA_GraDe(X,Ws,Gf[[2]],gf[[2]],dgf[[2]],b=0.998)$W, error=function(e) matrix(1,P,P))
    W5 <- tryCatch(JADE_GraDe(X,Ws,b=0.2)$V, error=function(e) matrix(1,P,P))
    W6 <- tryCatch(fICA(X,method = "sym2",g=gf[[2]],dg=dgf[[2]],G=Gf[[2]],n.init=2)$W, error=function(e) matrix(1,P,P))
    
    MDs4[i,l,1] <- MD(W1,B)
    MDs4[i,l,2] <- MD(W2,B)
    MDs4[i,l,3] <- MD(W3,B)
    MDs4[i,l,4] <- MD(W4,B)
    MDs4[i,l,5] <- MD(W5,B)
    MDs4[i,l,6] <- MD(W6,B)
    if((i/100)==floor(i/100)) print(paste("i=",i))
  }
}

write(MDs4,file="model4_data.dat")
