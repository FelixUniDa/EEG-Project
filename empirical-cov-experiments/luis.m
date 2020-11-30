%%
% This is the main file to run the tests. If you don't want to run all
% the different algorithms, set the options you want in this section, run
% it, then run the section corresponding to the algorithm you care about.
% Each section of this code will call singlecomparison() to get plot data
% for the algorithm of choice. Refer to singlecomparison.m for more
% details.

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 6;
lowerlimit = 500;
upperlimit = 1000;
step = 500;
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1}';

seed = 42;

numberofruns = 3;
sizes = lowerlimit:step:upperlimit;

addpath('third_party')

%% -------------------- FastICA with pow3 nonlinearity --------------------

% Set up empy arrays to hold the data
amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));
amaridatadampenedoracle = zeros(numberofruns,length(sizes));
frobeniusdatadampenedoracle = zeros(numberofruns,length(sizes));
amaridatadampenedcovariance = zeros(numberofruns,length(sizes));
frobeniusdatadampenedcovariance = zeros(numberofruns,length(sizes));
amaridatadampenedcentroid = zeros(numberofruns,length(sizes));
frobeniusdatadampenedcentroid = zeros(numberofruns,length(sizes));

% Run the comparison algorithm to get the data, storing it 
% each time
for i = 1:numberofruns
    % First, without dampening
    disp(['---Starting run ' int2str(i) '--------------']);
    rng(seed+i);
    s = rng;
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'seed', seed, ...
        'exponents', exponents, ...       
        'run', i, ...
        'verbose', 'true', 'orthogonalmix', 'false');
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    % With dampening, oracle
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', ...
        'regenerate_samples', 'false', ...
        'orthogonalmix', 'false', ...
        'seed', seed, ...
        'exponents', exponents, ...       
        'orthmethod', 'oracle');
    amaridatadampenedoracle(i,:) = result(1,:);
    frobeniusdatadampenedoracle(i,:) = result(2,:);

    % With dampening, covariance
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', 'regenerate_samples', 'false', ...
        'seed', seed, ...       
        'exponents', exponents, ...       
        'orthogonalmix', 'false', 'orthmethod', 'covariance');
    amaridatadampenedcovariance(i,:) = result(1,:);
    frobeniusdatadampenedcovariance(i,:) = result(2,:);

    % With dampening, centroid
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', 'regenerate_samples', 'false', ...
        'seed', seed, ...
        'exponents', exponents, ...       
        'orthogonalmix', 'false', 'orthmethod', 'centroid');
    amaridatadampenedcentroid(i,:) = result(1,:);
    frobeniusdatadampenedcentroid(i,:) = result(2,:);

end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedoracle, frobeniusdatadampenedoracle, sizes, 'FastICA - pow3 - oracle')

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedcovariance, frobeniusdatadampenedcovariance, sizes, 'FastICA - pow3 - covariance')

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedcentroid, frobeniusdatadampenedcentroid, sizes, 'FastICA - pow3 - centroid')
