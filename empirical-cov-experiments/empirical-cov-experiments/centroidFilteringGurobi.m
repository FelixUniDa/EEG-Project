function Y = centroidFilteringGurobi(X)

% centroid body  = (1/n) sum [-x_i, x_i]
% l*p = sum (1/n) lambda_i x_i, lambda_i in [-1,1]
[dim, n] = size(X);

minkowski = zeros(1,n);

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


beq = zeros(dim, 1);
beq(1:dim,:) = 0;
Aeq = sparse(dim, n+1);
Aeq(1:dim,lambdas) = X;


clear params;
clear model;
params.outputflag = 0;
    
%clear model;
model.obj = f;
model.rhs = beq;
model.lb = lb;
model.ub = ub;
model.sense = '=';
model.modelsense = 'min';

for i = 1:n
    
    p = X(:,i);
    Aeq(1:dim,l) = - n * p;
    model.A = Aeq;

    result = gurobi(model,params);
    minkowski(i) = -1/result.objval;

    model.vbasis = result.vbasis;
    model.cbasis = result.cbasis;
   
end

threshold = prctile(minkowski,75);

Y = X(:,minkowski <= threshold);
