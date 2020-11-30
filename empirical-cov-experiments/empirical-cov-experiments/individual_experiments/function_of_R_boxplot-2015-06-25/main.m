%% Plotting the error as a function of R
%

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 1000;
upperlimit = 5000;
step = 1000;
% exponents = '{2.1,2.1}';
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6,6}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,6,6,6,6,6}';
% exponents = '{2.1,2.1,6,6,6,6,6,6,6,6}';
% exponents = '{2.1,2.4,2.7,3.0,3.3,3.6,3.9,4.2,4.5,4.8}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,5}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,4}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,3}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.8}';
% exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.4}';
% exponents = 'Table[x, {x, 2.1, 5, (5-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 4, (4-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 3, (3-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 2.5, (2.5-2.1)/(dim - 1)}]';

orthogonalmix = true;
seed = 35;
rng(seed);

Rarr = 10:100:1010;

algorithm = 'pow3';

numberofruns = 10;
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


    
for k = 1:length(Rarr)
    error = zeros(2,length(sizes), numberofruns);
    
    for j = 1:numberofruns
        disp(['------ Starting run ' num2str(j)]);
        generatesamples(n, lowerlimit, upperlimit, step, ...
            'exponents', exponents, ...
            'seed', seed + j);
        
        for i=1:length(sizes)
            % Samples are m-by-n
            S = csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);

            % X is n-by-m
            X = A * S';
            [n,m] = size(X);

            % Plain FastICA
            [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
            Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
            [~, Aest1] = basisEvaluation(A, Aest1);

            % error(1,j) = norm(A - Aest1, 'fro');
            error(1,i,j) = norm(A-Aest1, 'fro');

            % Only Damping, then FastICA
            Xdamp = damp(X,Rarr(k));
            [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
            Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
            [~, Aest2] = basisEvaluation(A, Aest2);

            % error(2,j) = norm(A - Aest2, 'fro');
            error(2,i,j) = norm(A-Aest2, 'fro');

            if false
            % Only GIICA
            [~, Aest3, ~] = GIICA(X, 'whiten', 'verbose', 0);
            Aest3 = inv(Aest3);
            Aest3 = Aest3*(inv(diag(rownorm(Aest3'))));
            [~, Aest3] = basisEvaluation(A, Aest3);

            error(3,i,j) = norm(A-Aest3, 'fro');

            % Only Damping, GIICA
            Xdamp = damp(X,Rarr(k));
            [~, Aest4, ~] = GIICA(Xdamp, 'whiten', 'verbose', 0);
            Aest4 = inv(Aest4);
            Aest4 = Aest4*(inv(diag(rownorm(Aest4'))));
            [~, Aest4] = basisEvaluation(A, Aest4);

            error(4,i,j) = norm(A-Aest4, 'fro');
            end
        end
    end
    
    figure();
    set(gcf,'numbertitle','off','name', ['R = ' num2str(Rarr(k))]);
    subplot(1,2,1);
    plotdata = reshape(error(1,:,:),length(sizes), numberofruns);
    boxplot(plotdata');
    ax1 = gca;
    title('Plain FastICA');

    subplot(1,2,2);
    plotdata = reshape(error(2,:,:),length(sizes), numberofruns);
    boxplot(plotdata');
    ax2 = gca;
    title('Damping then FastICA');

    if false
    subplot(2,2,3);
    plotdata = reshape(error(3,:,:),length(sizes), numberofruns);
    boxplot(plotdata');
    ax3 = gca;
    title('GIICA');

    subplot(2,2,4);
    plotdata = reshape(error(4,:,:),length(sizes), numberofruns);
    boxplot(plotdata');
    ax4 = gca;
    title('Damping then GIICA');
    end

    ax1.YLim(1) = 0;
    ax2.YLim(1) = 0;
%         ax3.YLim(1) = 0;
%         ax4.YLim(1) = 0;

    plotmax = max([ax1.YLim(2), ax2.YLim(2)]);

    ax1.YLim(2) = plotmax;
%         ax3.YLim(2) = plotmax;

    ax2.YLim(2) = plotmax;
%         ax4.YLim(2) = plotmax;
end
