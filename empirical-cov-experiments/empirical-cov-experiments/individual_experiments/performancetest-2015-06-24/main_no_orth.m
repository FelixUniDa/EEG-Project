%% PERFORMANCE TEST
%
% This file plots the average error for estimated columns of A as a bar
% graph, each bar representing the average error of that column of A. Each
% plot is saved to the figures/ folder.
%
% COMMENTS:
% It seems that damping helps specifically on the columns which are *not*
% heavy-tailed.

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 5000;
upperlimit = 5000;
step = 5000;

% exponents = '{2.1,2.1}';
exponents = {};
exponents{1} = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
% exponents{2} = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6}';
% exponents{3} = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6,6}';
% exponents{1} = '{2.1,6,6,6,6,6,6,6,6,6}';
% exponents{2} = '{6,6,6,6,6,6,6,6,6,6}';

orthogonalmix = true;
seed = 54;
rng(seed);

% Rarr = 5:10:104;
Rarr = 5:1:5;

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

for k=1:length(exponents)
    disp(['Running for exponents ' exponents{k}]);
    for l=1:length(Rarr)
        for i=1:length(sizes)
            error = zeros(2,n,numberofruns);
            
            for j = 1:numberofruns
                disp(['--- Starting run ' num2str(j) ' -------'])
                generatesamples(n, sizes(i), sizes(i), step, ...
                    'exponents', exponents{k}, ...
                    'seed', seed);
                
                % m-by-n
                S = csvread(['../../samples/sample-' ...
                    num2str(sizes(i)) '.csv']);
                
                % X is n-by-m
                X = A * S';
                [n,m] = size(X);
                
                % Plain FastICA
                [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
%                 [~,Aest1inv,~] = GIICA(X, 'none');
%                 [~,Aest1inv,~] = GIICA(X,'whiten');
%                 Aest1 = inv(Aest1inv);
                Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
                [~, Aest1] = basisEvaluation(A, Aest1);
                
                error(1,:,j) = sqrt(sum((A-Aest1).^2,1));
                
                % Only Damping
                Xdamp = damp(X,Rarr(l));
                [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, ...
                    'verbose', 'off');
%                 [~,Aest2inv,~] = GIICA(Xdamp, 'none');
%                 [~,Aest2inv,~] = GIICA(Xdamp,'whiten');
%                 Aest2 = inv(Aest2inv);
                Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
                [~, Aest2] = basisEvaluation(A, Aest2);
                
                error(2,:,j) = sqrt(sum((A-Aest2).^2,1));
            end
            
            title(['FastICA - pow3 - R = ' num2str(Rarr(l)) ' - ' ...
                num2str(sizes(i)) ' Samples']);
            
            tmperror = mean(error,3);
            
            y = [tmperror(1,:)', tmperror(2,:)'];
            bar(y);
            legend('FastICA', 'FastICA with Damping');
            
            expstring = exponents{k};
            expstring = expstring(2:end-1);
            filename = ['figures/R' num2str(Rarr(l)) 'Exp' expstring];
            savefig([filename '.fig']);
            print([filename '.png'],'-dpng')
        end
    end
end

