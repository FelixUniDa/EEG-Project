%%
% This is the main file to run the tests. If you don't want to run all
% the different algorithms, set the options you want in this section, run
% it, then run the section corresponding to the algorithm you care about.
% Each section of this code will call singlecomparison() to get plot data
% for the algorithm of choice. Refer to singlecomparison.m for more
% details.

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 10;
lowerlimit = 100;
upperlimit = 5100;
step = 500;
% exponents = '{6,6,6,6,6,6,6,6,2.1,2.1}';
exponents ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';

seed = 452;
rng(seed);

algorithm = 'pow3';
R = 100;

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');
tic

%% ------------------------------ Experiments -----------------------------

% We have to pad the strings because matlab is awful...
% orthmethods = cellstr(['covariance'; 'centroid  '; ...
%                         'oracle    '; 'identity  ']);
orthmethods = {};
orthmethods{2} = 'identity';
orthmethods{1} = 'covariance';
orthmethods{3} = 'identity';

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, 3, length(sizes));

A = eye(n);

% Run the comparison algorithm to get the data, storing it each time
for i = 1:numberofruns
    disp(['------------- Starting run ' int2str(i) ' ------------------']);
    s = rng;

    generatesamples(n, lowerlimit, upperlimit, step, ...
        'exponents', exponents, ...
        'seed', seed + i);
    
    parfor j=1:length(sizes)
       S = csvread(['../../samples/sample-' num2str(sizes(j)) '.csv']);
       
       X = A * S'; % n-by-m
       m = size(X,2);
       
       error = zeros(1,3);
       
       % FastICA, no damping or orthogonalization
       [~,Aest1,~] = fastica(X, 'numOfIC', n, 'verbose', 'off');
       Aest1 = normalizecols(Aest1);
       [~,Aest1] = basisEvaluation(A,Aest1);
       % plotdata(i,1,j) = norm(A-Aest1,'fro');
       error(1) = norm(A-Aest1,'fro');
       
       % FastICA, with damping, no orthogonalization
       [Xdamp,acceptance,~] = damp(X,R);
       [~,Aest2,~] = fastica(Xdamp, 'numOfIC', n, 'verbose', 'off');
       Aest2 = normalizecols(Aest2);
       [~,Aest2] = basisEvaluation(A,Aest2);
       % plotdata(i,2,j) = norm(A-Aest2,'fro');
       error(2) = norm(A-Aest2,'fro');
       
       % FastICA, with damping and covariance orthogonalization
       [Xdamp2,acceptance2,~] = damp(inv(sqrtm((1/m)*(X*X')))*X,R);
       [~,Aest3,~] = fastica(Xdamp2, 'numOfIC', n, 'verbose', 'off');
       Aest3 = normalizecols(Aest3);
       [~,Aest3] = basisEvaluation(A,Aest3);
       % plotdata(i,3,j) = norm(A-Aest3,'fro');
       error(3) = norm(A-Aest3,'fro');
       
       plotdata(i,:,j) = error;
    end
end

meandata = mean(plotdata,1);

figure()
hold on;
plot(sizes, reshape(meandata(1,1,:), size(sizes)), '-or');
plot(sizes, reshape(meandata(1,2,:), size(sizes)), '-db');
plot(sizes, reshape(meandata(1,3,:), size(sizes)), '-*g');
% plot(sizes, reshape(meandata(1,4,:), size(sizes)), '-+k');

% legend('covariance','centroid','oracle','identity');
legend('No Damping, No orth', 'Damping, no orth', 'Damping, Centroid');

title('FastICA - pow3')
xlabel('Sample Size')
ylabel('Frobenius Error')
 
filename = ['FastICA - pow3 - Centroid Damping - R ' num2str(R)];
savefig(filename);
print(filename, '-dpng');

save('plotdata2.mat', 'plotdata');

toc
diary off
