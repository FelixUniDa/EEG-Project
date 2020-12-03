%% PERFORMANCE TEST
%
% This file plots the average error for estimated columns of A as a bar
% graph, each bar representing the average error of that column of A. Each
% plot is saved to the figures/ folder.
%
% COMMENTS:
% It seems that damping helps specifically on the columns which are *not*
% heavy-tailed.
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
% exponents = '{2.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5}';
orthogonalmix = true;
seed = 352;
rng(seed);

R = 10;

algorithm = 'pow3';

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

do_centroid = true;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% Experiment code

% A = eye(n);
A = normalizecols(mvnrnd(zeros(1,n), eye(n), n));

for i=1:length(sizes)
    error = zeros(6,n,numberofruns);
    
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
        
        % Plain FastICA
        [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
        Aest1 = normalizecols(Aest1);
        [~, Aest1] = basisEvaluation(A, Aest1);
        
        error(1,:,j) = sqrt(sum((A-Aest1).^2,1));
        
        % Only Damping - this is slightly nonsensical when A is not
        % orthogonal
        Xdamp = damp(X,R);
        [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
        Aest2 = normalizecols(Aest2);
        [~, Aest2] = basisEvaluation(A, Aest2);
        
        error(2,:,j) = sqrt(sum((A-Aest2).^2,1));
        
        if do_centroid
            % Centroid Error
            orthogonalizer = centroidOrthogonalizer(X);
            Xorthdamp = damp(orthogonalizer * X, R);
            [~, Aest3, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
            Aest3 = normalizecols(Aest3);
            [~, Aest3] = basisEvaluation(A, Aest3);
            Aest3 = normalizecols(inv(orthogonalizer)*Aest3);
            
            error(3,:,j) = sqrt(sum((A-Aest3).^2,1));
            
            % Centroid Error with "soft max"
            orthogonalizer = centroidOrthogonalizer(X, 'scale');
            Xorthdamp = damp(orthogonalizer * X, R);
            [~, Aest4, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
            Aest4 = normalizecols(Aest4);
            [~, Aest4] = basisEvaluation(A, Aest4);
            Aest4 = normalizecols(inv(orthogonalizer)*Aest4);
            
            error(4,:,j) = sqrt(sum((A-Aest4).^2,1));
        end
        
        % Covariance Orth plus Damping
        orthogonalizer = inv(sqrtm((1/m)*(X*X')));
        Xorthdamp = damp(orthogonalizer * X, R);
        [~, Aest5, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
        Aest5 = normalizecols(Aest5);
        [~, Aest5] = basisEvaluation(A, Aest5);
        Aest5 = normalizecols(inv(orthogonalizer)*Aest5);

        error(5,:,j) = sqrt(sum((A-Aest5).^2,1));
        
        % Oracle Orth plus Damping
        orthogonalizer = orth(A)*inv(A);
        Xorthdamp = damp(orthogonalizer * X, R);
        [~, Aest6, ~] = fastica(Xorthdamp, 'numOfIC', n, 'verbose', 'off');
        Aest6 = normalizecols(Aest6);
        [~, Aest6] = basisEvaluation(A, Aest6);
        Aest6 = normalizecols(inv(orthogonalizer)*Aest6);

        error(6,:,j) = sqrt(sum((A-Aest6).^2,1));
    end
    
    figure();
    hold on;
    
    title(['FastICA - pow3 - R = ' num2str(R)]);
    
    % Order by cumulants
    % proj = X'*A;
    % fourth = sum(proj.^4,1)/m;
    % second = sum(proj.^2,1)/m;
    
    meandata = mean(error,3);
    
    if do_centroid
        y = [meandata(1,:)', meandata(2,:)', meandata(3,:)', meandata(4,:)', meandata(5,:)', meandata(6,:)'];
    else
        y = [meandata(1,:)',meandata(2,:)',meandata(5,:)', meandata(6,:)'];
    end
    
    bar(y);
    
    if do_centroid
        legend('Plain FastICA', 'Only Damping', 'Centroid', 'Centroid With Scaling', 'Covariance', 'Oracle');
    else
        legend('Plain FastICA', 'Only Damping', 'Covariance', 'Oracle');
    end
end

