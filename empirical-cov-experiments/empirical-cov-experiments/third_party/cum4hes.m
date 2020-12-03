% cum4hes(X, u, XXt)
% H -- Returned hessian matrix.
% X -- Matrix where each column represents a data point.  The data is
%      assumed to be centered (i.e., mean subtracted).
% u -- directional vector.  The Hessian of the u-directional fourth cumulant
%      is taken.
% XXt -- optional input argument.  If provided, then the calculation of
%        X*X' can be avoided.  This is used for efficiency purposes when
%        the function is called multiple times.
% Computes and returns an approximation to the hessian with respect to the
% directional variable u of k_4(u'x) where x denotes the
% random variable being sampled.

function H = cum4hes(X, u, XXt)
    if ~exist('XXt', 'var')
        XXt = X*X';
    end
    %% compute the unbiased estimator
    [d, n] = size(X);
    tM = repmat((u'*X), d, 1).*X;
    H = 12*(tM*tM') * (n^2*(n+1))/(n*(n-1)*(n-2)*(n-3));

    tM = XXt*u / n;
    H = H - 24*(tM*tM') * n^2*(n-1)/((n-1)*(n-2)*(n-3));
    H = H - 12*(sum((u'*X).^2) / n) * ((XXt) / n) * n^2*(n-1)/((n-1)*(n-2)*(n-3));
end
