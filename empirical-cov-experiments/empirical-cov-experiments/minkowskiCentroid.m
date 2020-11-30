function minkowski = minkowskiCentroid(X, p)

% centroid body  = (1/n) sum [-x_i, x_i]
% l*p = sum (1/n) lambda_i x_i, lambda_i in [-1,1]
[dim, n] = size(X);

% assert(size(p) == [dim 1]);
lambdas = (1:n)';
l = n+1;
vars = [lambdas; l];

% maximize l
f = zeros(size(vars));
f(l) = -1;

%s.t.
% -1 <= lambdas <= 1
% (1/n) X * lambdas = l * p 
%   (i.e. X*lambdas - n l * p = 0

lb = zeros(size(vars));
ub = zeros(size(vars));
lb(lambdas,:) = -ones(size(lambdas));
ub(lambdas,:) = ones(size(lambdas));
ub(l,:) = Inf;
A = [];
b = [];
beq = zeros(dim, 1);
beq(1:dim,:) = 0;
Aeq = sparse(dim, n+1);
Aeq(1:dim,lambdas) = X;
Aeq(1:dim,l) = - n * p;
%options = optimoptions(@linprog, 'Display','none','Algorithm', 'simplex');
options = optimoptions(@linprog, 'Display','none');
[~, fval] = linprog(f, A, b, Aeq, beq, lb, ub, [], options);

minkowski = -1/fval;