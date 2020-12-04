function orthogonalizer = centroidOrthogonalizer(X, method)

% [dim, n] = size(X);

if nargin < 2
    method = 'filter';
end
switch method
    case 'scale'
        minkowski = minkowskiCentroidGurobi(X,X);
        scaling = diag(tanh(minkowski) ./ minkowski);
        sample = X * scaling;
    case 'filter'
        %sample = centroidFiltering(X);
        sample = centroidFilteringGurobi(X);
end

samplesize = size(sample,2);

C = sample * sample' / samplesize;

orthogonalizer = inv(sqrtm(C));

end

