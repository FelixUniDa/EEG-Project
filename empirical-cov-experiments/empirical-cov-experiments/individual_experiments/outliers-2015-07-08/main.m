%% Plotting the error as a function of R
%
% This experiment is a sanity check, to study the effect of far outliers on
% FastICA. It generates samples uniformly from the hypercube, then adds an
% extreme outlier along the first dimension, simulating the case where one
% component heavy tailed and either the others are light tailed or they are
% heavy tailed and, by luck, don't have outliers as extreme as the first.
%
% COMMENTS:
% In general, the columns of A *with* outliers are recocvered much more
% accurately than when they do not have outliers, but in this case, the
% "short" components have error up to several magnitudes larger.
%
% Ex. In two dimensions:
% Error on each column, without outliers: 
%      [0.0055104487241238 0.00458430601068905]
% Error on each column, with outliers: 
%      [4.1132876252527e-06 0.011650017557223]

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 2;
lowerlimit = 5000;
upperlimit = 5000;
step = 2000;

seed = 35;
rng(seed);

Rarr = 5:2:100;

algorithm = 'pow3';

numberofruns = 1;
sizes = lowerlimit:step:upperlimit;

% Decide whether or not to make boxplots
do_boxplot = false;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% ------------- Begin Experiment -------------

A = eye(n);

for i=1:length(sizes)
    error = zeros(2, numberofruns, n);
    for j=1:numberofruns
        S = unifrnd(-1,1,n,sizes(i));
        S2 = [S [1000 zeros(1,n-1)]' [2000 zeros(1,n-1)]'];
        
        [~,Aest1,~] = fastica(S,'verbose','off');
        Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
        [~, Aest1] = basisEvaluation(A, Aest1);
%         error(1,j) = norm(A-Aest1, 'fro');
        error(1,j,:) = sum((A-Aest1).^2,1).^(1/2);
        
%         disp(['Error on each column, without outliers: ' ...
%             mat2str( ]);
        
        [~,Aest2,~] = fastica(S2,'verbose','off');
        Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
        [~, Aest2] = basisEvaluation(A, Aest2);
        error(2,j,:) = sum((A-Aest1).^2,1).^(1/2);
        
%         disp(['Error on each column, with outliers: ' ...
%             mat2str(sum((A-Aest2).^2,1).^(1/2)) ]);
    end
    
    meanerror = mean(error,2);
    
    bar(meanerror');
end