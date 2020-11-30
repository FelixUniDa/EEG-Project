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
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
orthogonalmix = true;
seed = 352;
rng(seed);

R = 100;

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
    error1 = zeros(1,n);
    error2 = zeros(1,n);
    error3 = zeros(1,n);
    cumulants = zeros(1,n);
    
    for j = 1:numberofruns
        disp(['--- Starting run ' num2str(j) ' -------'])
        generatesamples(n, sizes(i), sizes(i), step, ...
            'exponents', exponents, ...
            'seed', seed);

        % m-by-n
        S = csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);

        % X is n-by-m
        X = A * S';
        [n,m] = size(X);
        
        % Calculate the average fourth cumulant
        % proj = X'*A;
        % Estimate fourth and second moment, then combine to get cumulant
        % fourth = sum(proj.^4,1)/m;
        % second = sum(proj.^2,1)/m;
        
        % cumulants = ((fourth - 3*(second.^2)) + (j-1)*cumulants)/j;

        [~, Aest1, ~] = fastica(X, 'numOfIC', n);
        Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
        [~, Aest1] = basisEvaluation(A, Aest1);
        
        % Calculate the running mean
        error1 = (sqrt(sum((A-Aest1).^2,1)) + (j-1)*error1)/j;

        Xdamp = damp(X,R);
        [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n);
        Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
        [~, Aest2] = basisEvaluation(A, Aest2);
        
        error2 = (sqrt(sum((A-Aest2).^2,1)) + (j-1)*error2)/j;

        orthogonalizer = centroidOrthogonalizer(X);
        Xorthdamp = damp(orthogonalizer * X, R);
        [~, Aest3, ~] = fastica(Xorthdamp, 'numOfIC', n);
        Aest3 = Aest3*(inv(diag(rownorm(Aest3'))));
        [~, Aest3] = basisEvaluation(A, Aest3);
        
        error3 = (sqrt(sum((A-Aest3).^2,1)) + (j-1)*error3)/j;
    end
    
    figure();
    hold on;

    title(['FastICA - pow3 - R = ' num2str(R)]);

    % Order by cumulants
    proj = X'*A;
    fourth = sum(proj.^4,1)/m;
    second = sum(proj.^2,1)/m;

    cumulants = fourth - 3*(second.^2)
    [~,ind] = sort(cumulants);
    error1 = error1(ind);
    error2 = error2(ind);
    error3 = error3(ind);

    y = [error1', error2', error3'];
    bar(y);
    legend('Plain FastICA', 'Only Damping', 'Centroid');
end

