%% 
% This file plots the average error for estimated columns of A, along with
% the fourth cumulant of the sample in the direction of the column
%

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 5000;
upperlimit = 5000;
step = 5000;
% exponents = '{2.1,2.1}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
exponents = '{6,6,6,6,6,6,2.1,2.1,2.1,2.1}';
orthogonalmix = true;
seed = 352;
rng(seed);

Rarr = 50:10:100;

algorithm = 'pow3';

numberofruns = 1;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% Experiment code

A = eye(n);
% A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
% A = A*(inv(diag(rownorm(A'))));

for i=1:length(sizes)
    error = zeros(4,n);
    
    for j = 1:length(Rarr)
        generatesamples(n, sizes(i), sizes(i), step, ...
            'exponents', exponents, ...
            'seed', seed);

        % m-by-n
        S = csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);

        % X is n-by-m
        X = A * S';
        [n,m] = size(X);

        % Plain FastICA
        [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
        Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
        [~, Aest1] = basisEvaluation(A, Aest1);
        
        % Calculate the running mean
        error(1,:) = (sqrt(sum((A-Aest1).^2,1)) + (j-1)*error(1,:))/j;

        % Only Damping
        Xdamp = damp(X,Rarr(j));
        [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
        Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
        [~, Aest2] = basisEvaluation(A, Aest2);
        
        % Calculate the running mean
        error(2,:) = (sqrt(sum((A-Aest2).^2,1)) + (j-1)*error(2,:))/j;

        % Centroid Error
        orthogonalizer = centroidOrthogonalizer(X);
        Xorthdamp = damp(orthogonalizer * X, Rarr(j));
        [~, Aest3, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
        Aest3 = Aest3*(inv(diag(rownorm(Aest3'))));
        [~, Aest3] = basisEvaluation(A, Aest3);
        
        % Calculate the running mean
        error(3,:) = (sqrt(sum((A-Aest3).^2,1)) + (j-1)*error(3,:))/j;
        
        % Centroid Error with "soft max"
        orthogonalizer = centroidOrthogonalizer(X, 'scale');
        Xorthdamp = damp(orthogonalizer * X, Rarr(j));
        [~, Aest4, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
        Aest4 = Aest4*(inv(diag(rownorm(Aest4'))));
        [~, Aest4] = basisEvaluation(A, Aest4);
        
        % Calculate the running mean
        error(4,:) = (sqrt(sum((A-Aest4).^2,1)) + (j-1)*error(4,:))/j;
        
        figure();
        hold on;

        title(['FastICA - pow3 - R ' num2str(Rarr(j))]);

        % Order by cumulants
        % proj = X'*A;
        % fourth = sum(proj.^4,1)/m;
        % second = sum(proj.^2,1)/m;

        y = error';
        bar(y);
        legend('Plain FastICA', 'Only Damping', 'Centroid', 'Centroid With Scaling');
    end
end

