function [gamma] = gammaest(X)
%% Function alphaest
%  Takes input X, vector or matrix

    X(X==0) = 0.0000000001;
    Z = log(abs(X)); % new log of the stochastic process
    
    sz = size(Z);
    
    if sz(1) == 1
        Z_mean = mean(Z);
    else
        Z_mean = mean(Z,2);
    end
    
    alpha = alphaest(X);
    
    eulergamma = 0.5772156649;
    
    gamma = exp( ...
        alpha .* ...
            (Z_mean - eulergamma*(alpha.^(-1) - ones(size(alpha))))...
        );
end