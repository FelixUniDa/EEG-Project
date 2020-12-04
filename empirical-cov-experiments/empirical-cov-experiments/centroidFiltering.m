function Y = centroidFiltering(X)

[dim, n] = size(X);

minkowski = zeros(1,n);
for i = 1:n
   %minkowski(i) = minkowskiCentroid(X, X(:,i));
   minkowski(i) = minkowskiCentroidGurobi(X, X(:,i));
%   disp('.');
end

threshold = prctile(minkowski,75);

Y = X(:,minkowski <= threshold);
