function [B] = normalizecols(A)

B = A*(inv(diag(rownorm(A'))));
