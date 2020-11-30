%% Plotting the error as a function of R
%
% Plots error as a function of R, one plot for each string of exponents in
% the cell called `exponents` set up in the first code section.
%
% COMMENTS:
% This experiment is meant to illustrate the difference between the error
% computed as either the frobenius norm of the difference of A and \hat{A}
% or the sum of the distances between columns. Currently, there seems to be
% no remarkable difference, other than that the latter is some constant
% larger than the former.

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 100;
upperlimit = 5100;
step = 1000;
% exponents = '{2.1,2.1}';
exponents = {};
exponents{1} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
exponents{2} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6}';
exponents{3} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6,6}';
exponents{4} ='{2.1,2.1,2.1,2.1,2.1,6,6,6,6,6}';
exponents{5} ='{2.1,2.1,6,6,6,6,6,6,6,6}';
exponents{6} ='{2.1,6,6,6,6,6,6,6,6,6}';
exponents{7} ='{6,6,6,6,6,6,6,6,6,6}';
exponents{8} ='{2.1,2.4,2.7,3.0,3.3,3.6,3.9,4.2,4.5,4.8}';
exponents{9} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,5}';
exponents{10} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,4}';
exponents{11} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,3}';
exponents{12} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.8}';
exponents{13} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.6}';
exponents{14} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.4}';
% exponents = 'Table[x, {x, 2.1, 5, (5-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 4, (4-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 3, (3-2.1)/(dim - 1)}]';
% exponents = 'Table[x, {x, 2.1, 2.5, (2.5-2.1)/(dim - 1)}]';

exponents = [exponents; {'{2.5,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}'}];
exponents = [exponents; {'{2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5}'}];
exponents = [exponents; {'{3,3,3,3,3,3,3,3,3,3}'}];
exponents = [exponents; {'{3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5}'}];
exponents = [exponents; {'{4,4,4,4,4,4,4,4,4,4}'}];
exponents = [exponents; {'{4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25}'}];
exponents = [exponents; {'{4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5}'}];
exponents = [exponents; {'{5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1}'}];

orthogonalmix = true;
seed = 35;
rng(seed);

Rarr = 5:2:100;
R = 10;

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
    
    froberror = zeros(2,length(sizes), numberofruns);
    l1error = zeros(2,length(sizes), numberofruns);
    rejected = zeros(1,length(sizes));
    
    for j = 1:numberofruns
        disp(['------ Starting run ' num2str(j)]);
        generatesamples(n, lowerlimit, upperlimit, step, ...
            'exponents', exponents{l}, ...
            'seed', seed + j);
        
        for i=1:length(sizes)
            % Samples are m-by-n
            S=csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);
            
            % X is n-by-m
            X = A * S';
            [n,m] = size(X);
            
            % Plain FastICA
            [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
            Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
            [~, Aest1] = basisEvaluation(A, Aest1);
            
            % error(1,j) = norm(A - Aest1, 'fro');
            froberror(1,i,j) = norm(A-Aest1, 'fro');
            l1error(1,i,j) = sum(sqrt(sum((A-Aest1).^2,1)));
            
            % Only Damping, then FastICA
            [Xdamp, percentremaining] = damp(X,R);
            [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
            Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
            [~, Aest2] = basisEvaluation(A, Aest2);
            
            rejected(i) = percentremaining;
            
            % error(2,j) = norm(A - Aest2, 'fro');
            froberror(2,i,j) = norm(A-Aest2, 'fro');
            l1error(2,i,j) = sum(sqrt(sum((A-Aest2).^2,1)));
        end
    end
    
    myfig = figure(1);
    clf(myfig);
    set(myfig, 'Position', [100, 600, 800, 600]);
    hold on;
    
    title(['FastICA - pow3 - R=' num2str(R)]);
    xlabel('Sample Size');
    
    medianfrob = median(froberror,3);
    medianl1 = median(l1error,3);
    plot(sizes, medianfrob(1,:), '-*');
    plot(sizes, medianfrob(2,:), '-o');
    plot(sizes, medianl1(1,:), '-*');
    plot(sizes, medianl1(2,:), '-o');
    plot(sizes, rejected, '-.d');
    %         plot(Rarr, ones(1,length(Rarr)), '-');
    legend('FastICA Frobenius', 'FastICAm - Damped Frobenius', ...
        'FastICA L1', 'FastICAm - Damped L1', ...
        'Fraction retained');
    
    expstring = exponents{l};
    expstring = expstring(2:end-1);
    filename = ['figures/Exp' expstring];
    savefig([filename '.fig']);
    print([filename '.png'], '-dpng');
end

disp('Done!')

% if do_boxplot
%     figure();
%     set(gcf,'numbertitle','off','name',[num2str(sizes(i)) ' Samples']);
%     subplot(1,1);
%     plotdata = reshape(barerror(1,:,:),length(Rarr), numberofruns)';
%     boxplot(plotdata, 'labels', Rarr);
%     ax1 = gca;
%     title('Plain FastICA');
%
%     subplot(1,2,2);
%     plotdata = reshape(barerror(2,:,:),length(Rarr), numberofruns)';
%     boxplot(plotdata, 'labels', Rarr);
%     ax2 = gca;
%     title('Damping then FastICA');
%
%     ax1.YLim(1) = 0;
%     ax2.YLim(1) = 0;
%
%     plotmax = max([ax1.YLim(2),ax2.YLim(2)]);
%
%     ax1.YLim(2) = plotmax;
%     ax2.YLim(2) = plotmax;
% end
