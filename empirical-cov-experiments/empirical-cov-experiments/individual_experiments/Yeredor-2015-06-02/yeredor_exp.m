%%
% Centroid experiment in high dimensions
% 29 May 2015

warning('off','MATLAB:normest:notconverge');

n = 10;
lowerlimit = 1000;
upperlimit = 41000;
step = 5000;

%exponents = '{6,6,6,6,6,6,6,6,2.1,2.1}';
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
numberofruns = 10;

seed = 42;
rng(seed);

sizes = lowerlimit:step:upperlimit;

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
        'orthogonalmix','true','algorithm','yeredor','verbose', 'true', ...
        'exponents', exponents, ...
        'seed', seed, ...
        'run', i);
    amaridataundampened(i,:) = result(1,:);
    frobeniusdataundampened(i,:) = result(2,:);
    
   
end

myicaplot(amaridataundampened, frobeniusdataundampened, ...
    amaridatadampened, frobeniusdatadampened, sizes, 'Yeredor')