function [alpha] = alphaest(X)
%% Function alphaest
%  Takes input X, vector or matrix

    X(X==0) = 0.0000000001;
    X = log(abs(X));
    
    sz = size(X);
    
    if sz(1) == 1
        X_var = var(X,1);
    else
        X_var = var(X,1,2);
    end
    
    alpha = real((X_var*(6/(pi^2)) - (1/2)).^(-1/2));
end