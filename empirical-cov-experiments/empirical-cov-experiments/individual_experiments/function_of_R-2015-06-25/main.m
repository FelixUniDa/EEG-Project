%% Plotting the error as a function of R
%
% Plots the frobenius error of FastICA and FastICA with damping, as a
% function of R, one plot for each exponent string in the `exponents` cell,
% saving each into the figures/ folder.

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 5000;
upperlimit = 5000;
step = 2000;
% exponents = '{2.1,2.1}';
exponents = {};
exponents{1} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
% exponents{2} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6}';
% exponents{3} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6,6}';
% exponents{4} ='{2.1,2.1,2.1,2.1,2.1,6,6,6,6,6}';
% exponents{5} ='{2.1,2.1,6,6,6,6,6,6,6,6}';
% exponents{6} ='{2.1,6,6,6,6,6,6,6,6,6}';
% exponents{7} ='{6,6,6,6,6,6,6,6,6,6}';
% exponents{8} ='{2.1,2.4,2.7,3.0,3.3,3.6,3.9,4.2,4.5,4.8}';
% exponents{9} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,5}';
% exponents{10} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,4}';
% exponents{11} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,3}';
% exponents{12} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.8}';
% exponents{13} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.6}';
% exponents{14} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.4}';
% exponents = 'Table[x, {x, 2.1, 5, (5-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 4, (4-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 3, (3-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 2.5, (2.5-2.1)/(dim - 1)}]';

% exponents ='{1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9}';
% exponents = [exponents; {'{2.5,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}'}];
% exponents = [exponents; {'{2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5}'}];
% exponents = [exponents; {'{3,3,3,3,3,3,3,3,3,3}'}];
% exponents = [exponents; {'{3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5}'}];
% exponents = [exponents; {'{4,4,4,4,4,4,4,4,4,4}'}];
% exponents = [exponents; {'{4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25}'}];
% exponents = [exponents; {'{4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5}'}];
% exponents = [exponents; {'{5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1}'}];
% exponents ='{10,10,10,10,10,10,10,10,10,10}';

orthogonalmix = true;
seed = 35;
rng(seed);

Rarr = 5:2:100;

algorithm = 'pow3';

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

% Decide whether or not to make boxplots
do_boxplot = false;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% Experiment code

A = eye(n);
% A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
% A = A*(inv(diag(rownorm(A'))));

for l=1:length(exponents)
    disp(['--- Running for exponents ' exponents{l}]);
    
    for i=1:length(sizes)
        barerror = zeros(3,length(Rarr), numberofruns);
        rejected = zeros(1,length(Rarr));
        rejected2 = zeros(numberofruns,length(Rarr));
        
        for j = 1:numberofruns
            disp(['------ Starting run ' num2str(j)]);
            generatesamples(n, sizes(i), sizes(i), step, ...
                'exponents', exponents{l}, ...
                'seed', seed + j);
            % Samples are m-by-n
            S=csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);
            
            for k = 1:length(Rarr)
                % disp(['--------- Chosing R = ' num2str(Rarr(k))]);
                
                % X is n-by-m
                X = A * S';
                [n,m] = size(X);
                
                % Plain FastICA
                [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
                Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
                [~, Aest1] = basisEvaluation(A, Aest1);
                
                barerror(1,k,j) = norm(A-Aest1, 'fro');
                
                % Only Damping, then FastICA
                [Xdamp, percentremaining] = damp(X,Rarr(k));
                [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
                Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
                [~, Aest2] = basisEvaluation(A, Aest2);
                
                rejected(j,k) = percentremaining;
                barerror(2,k,j) = norm(A-Aest2, 'fro');
                
                % Covariance Damping, then FastICA
                B = inv(sqrtm((1/m)*(X*X')));
                [Xdamp, percentremaining] = damp(B*X,Rarr(k));
                [~, Aest3, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
                Aest3 = Aest3*(inv(diag(rownorm(Aest3'))));
                [~, Aest3] = basisEvaluation(A, Aest3);
                
                rejected2(j,k) = percentremaining;
                barerror(3,k,j) = norm(A-Aest3, 'fro');
            end
        end
        
        myfig = figure(1);
        clf(myfig);
        set(myfig, 'Position', [100, 600, 800, 600]);
        hold on;
        
        title(['FastICA - pow3 - ' num2str(sizes(i)) ' samples']);
        xlabel('R');
        
        meanerror = median(barerror,3);
        plot(Rarr, meanerror(1,:), '-*');
        plot(Rarr, meanerror(2,:), '-o');
        plot(Rarr, meanerror(3,:), '-s');
        plot(Rarr, mean(rejected,1), '-.d');
        plot(Rarr, mean(rejected2,1), '-.d');
        plot(Rarr, ones(1,length(Rarr)), '-');
        legend('FastICA', 'FastICAm - Damped', ...
            'FastICAm - Covariance Damped', ...
            'Fraction retained', ...
            'Covariance Fraction retained', ...
            '100% Acceptance');
        
        expstring = exponents{l};
        expstring = expstring(2:end-1);
        filename = ['figures/FastICA-' algorithm '-Exp' expstring];
        savefig([filename '.fig']);
        print([filename '.png'], '-dpng');
    end
end

disp('Done!');
