%%
% Experiment for asymmetrical distribution.
% Matlab code for density:
% 
% f[exp_][x_] := 1/(x^(exp));
% 
% norm[exp_] := NIntegrate[f[exp][Abs[x-1] + 1.5], {x, -Infinity, Infinity}];
% 
% P[exp_] := 
%   ProbabilityDistribution[
%    f[exp][Abs[x-1] + 1.5]/norm[exp], {x, -Infinity, Infinity}];

n = 10;
lowerlimit = 1000;
upperlimit = 51000;
step = 10000;

exponents = '{6,6,6,6,6,6,6,6,2.1,2.1}';

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

seed = 42;
rng(seed);

addpath('../../')
addpath('../../third_party')

%% -------------------- FastICA with pow3 nonlinearity --------------------

% Set up empy arrays to hold the data
amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));
amaridatadampened = zeros(numberofruns,length(sizes));
frobeniusdatadampened = zeros(numberofruns,length(sizes));

% Run the comparison algorithm to get the data, storing it 
% each time
for i = 1:numberofruns
    % First, without dampening
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    % With dampening
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', ...
        'regenerate_samples', 'false', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'FastICA - pow3')

%% -------------------- FastICA with tanh nonlinearity --------------------

amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));
amaridatadampened = zeros(numberofruns,length(sizes));
frobeniusdatadampened = zeros(numberofruns,length(sizes));

for i = 1:numberofruns
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'algorithm', 'tanh', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', 'algorithm', 'tanh', ...
        'regenerate_samples', 'false', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'FastICA - tanh')

%% -------------------------- Fourier PCA ---------------------------------

amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));
amaridatadampened = zeros(numberofruns,length(sizes));
frobeniusdatadampened = zeros(numberofruns,length(sizes));

for i = 1:numberofruns
    % With dampening
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'algorithm', 'fpca', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    % Without dampening
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', 'algorithm', 'fpca', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'Fourier PCA')

%% -------------------------------- SOBI ----------------------------------

amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));
amaridatadampened = zeros(numberofruns,length(sizes));
frobeniusdatadampened = zeros(numberofruns,length(sizes));

for i = 1:numberofruns
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'algorithm', 'sobi', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', 'algorithm', 'sobi', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'SOBI')

%% -------------------------------- JADE ----------------------------------

amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));

for i = 1:numberofruns
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'algorithm', 'jade', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', 'algorithm', 'jade', ...
        'regenerate_samples', 'false', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'JADE')

%% -------------------------------- SIMPLE --------------------------------

amaridataundampened = zeros(numberofruns,length(sizes));
frobeniusdataundampened = zeros(numberofruns,length(sizes));

for i = 1:numberofruns
    disp(['Starting run ' int2str(i)]);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'algorithm', 'simple', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
        'verbose', 'true', 'dampen', 'true', 'algorithm', 'simple', ...
        'regenerate_samples', 'false', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridatadampened(i,:) = result(1,:);
    frobeniusdatadampened(i,:) = result(2,:);
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'Eigendecomp')