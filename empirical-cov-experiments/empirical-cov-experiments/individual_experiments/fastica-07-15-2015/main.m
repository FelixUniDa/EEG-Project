%% Behavior of fastica with damping and without damping
% Without damping, in dim=2,for an array of pairs of exponents greater than 1
%(say, exponents (1+i/10, 1+j/10) for i,j from 1 to some large value)
% construct the array of resulting median of the error of many trials,
% for each pair of exponents. Then plot using something like mathematica's
% arrayplot or matrix plot. Then maybe the same for fastica with damping.

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 2;
lowerlimit = 10000;
upperlimit = 10000;
step = 10000;

sizes = 10000;

% exponents = '{2.1,2.1}';
exponents = {};

for i=1:30
    for j=1:30
        e = ['{' num2str(1+ i/10.0) ',' num2str(1+ j/10.0) '}'];
        exponents = [exponents; e];
    end
end

orthogonalmix = true;
seed = 35;
rng(seed);

%Rarr = 5:2:100;
R = 50;

algorithm = 'pow3';

numberofruns = 40;
%sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% Experiment code

A = eye(n);
% A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
% A = A*(inv(diag(rownorm(A'))));
 froberror = zeros(2,length(exponents), numberofruns);
 %l1error = zeros(2,length(exponents), numberofruns);
   
for i=1:length(exponents)
    disp(['--- Running for exponents ' exponents{i}]);
    
    %rejected = zeros(1,numberofruns);
    
    for j = 1:numberofruns
        disp(['------ Starting run ' num2str(j)]);
        generatesamples(n, lowerlimit, upperlimit, step, ...
            'exponents', exponents{i}, ...
            'seed', seed +100+ j);
        
        
        % Samples are m-by-n
        S=csvread(['../../samples/sample-' num2str(sizes) '.csv']);
        
        % X is n-by-m
        X = A * S';
        [n,m] = size(X);
        
        % Plain FastICA
        try
            
            [~, Aest1, ~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
            Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
            [~, Aest1] = basisEvaluation(A, Aest1);
        catch ME
            warning('Error in undamped Fast ICA in exponent %s and run %d',exponents{i},j);
            Aest1 = NaN(n);
            %disp(ME.message);
        end
            
        % error(1,j) = norm(A - Aest1, 'fro');
        froberror(1,i,j) = norm(A-Aest1, 'fro');
        %l1error(1,i,j) = sum(sqrt(sum((A-Aest1).^2,1)));
        
        % Only Damping, then FastICA
        [Xdamp, percentremaining] = damp(X,R);
        [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
        Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
        [~, Aest2] = basisEvaluation(A, Aest2);
        
        %rejected(i) = percentremaining
        
        % error(2,j) = norm(A - Aest2, 'fro');
        froberror(2,i,j) = norm(A-Aest2, 'fro');
        %l1error(2,i,j) = sum(sqrt(sum((A-Aest2).^2,1)));
    end
end


medianfrob = median(froberror,3)
%medianl1 = median(l1error,3);

% 
% expstring = exponents{l};
% expstring = expstring(2:end-1);
filename = 'figures/datatanh.mat';
save(filename,'-v4','medianfrob');

% figname = ['figures/',expstring];
disp('Done!')

% thisfolder = strrep(mfilename('fullpath'), mfilename(), '');
% 
% [~,l] = size(medianfrob);
% params =  [filename ' ' int2str(l) ' ' figname];
% 
% if ispc
%     cmd = ['"C:\Program Files\Wolfram Research\Mathematica\10.1\math" -script "' ...
%      thisfolder 'plot.m" ' param];
% else
%     cmd = [thisfolder 'mathematicasamples.w ' params];
% end
% 
% system(cmd);
% 
% 
