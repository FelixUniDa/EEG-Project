%% File: main.m
% Centroid experiment in high dimensions with the same exponents
% 29 May 2015

warning('off','MATLAB:normest:notconverge');

n = 10;
lowerlimit = 1000;
upperlimit = 11000;
step = 1000;
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
numberofruns = 10;

seed = 42;
rng(seed);

sizes = lowerlimit:step:upperlimit;

% Make sure we can call the files that do the heavy lifting
addpath('../../')
addpath('../../third_party')

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
    s = rng;
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'orthogonalmix', 'false', ...
        'exponents', exponents);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    % With dampening, oracle
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', ...
        'regenerate_samples', 'false', ...
        'orthogonalmix', 'false', ...
        'orthmethod', 'oracle', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampenedoracle(i,:) = result(1,:);
    frobeniusdatadampenedoracle(i,:) = result(2,:);

    % With dampening, covariance
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', 'regenerate_samples', 'false', ...
        'orthogonalmix', 'false', 'orthmethod', 'covariance', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampenedcovariance(i,:) = result(1,:);
    frobeniusdatadampenedcovariance(i,:) = result(2,:);

    % With dampening, centroid
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'damp', 'true', 'regenerate_samples', 'false', ...
        'orthogonalmix', 'false', 'orthmethod', 'centroid', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampenedcentroid(i,:) = result(1,:);
    frobeniusdatadampenedcentroid(i,:) = result(2,:);

end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedoracle, frobeniusdatadampenedoracle, sizes, 'FastICA - pow3 - oracle')

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedcovariance, frobeniusdatadampenedcovariance, sizes, 'FastICA - pow3 - covariance')

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampenedcentroid, frobeniusdatadampenedcentroid, sizes, 'FastICA - pow3 - centroid')
