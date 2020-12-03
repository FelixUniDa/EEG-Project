%%
% This is the main file to run the tests. If you don't want to run all
% the different algorithms, set the options you want in this section, run
% it, then run the section corresponding to the algorithm you care about.
% Each section of this code will call singlecomparison() to get plot data
% for the algorithm of choice. Refer to singlecomparison.m for more
% details.

n = 10;
lowerlimit = 1000;
upperlimit = 11000;
step = 2000;
exponents = '{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';

seed = 42;
rng(seed);

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

diary('output.txt');
tic

%% ------------------------------ Experiments -----------------------------

algorithms = cellstr(['pow3   '; 'tanh   '; 'jade   '; 'yeredor'; 'fpca   ']);
% lines = cellstr(['']);

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, length(algorithms), length(sizes));
plotdatadampened = zeros(numberofruns, length(algorithms), length(sizes));

% Run the comparison algorithm to get the data, storing it each time
for i = 1:numberofruns
    disp(['Starting run ' int2str(i)]);

    for j = 1:length(algorithms)
        % First, without dampening
        result = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'true', ...
            'algorithm', algorithms{j}, ...
            'exponents', exponents, ...
            'seed', seed, ...
            'run', i);
        plotdata(i,j,:) = result(2,:);

        % With dampening
        result = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'true', ...
            'damp', 'true', ...
            'algorithm', algorithms{j}, ...
            'regenerate_samples', 'false', ...
            'exponents', exponents, ...
            'seed', seed, ...
            'run', i);
        plotdatadampened(i,j,:) = result(2,:);
    end
end

meandata = mean(plotdata,1);
meandatadampened = mean(plotdatadampened,1);

figure()
hold on;
for i=1:length(algorithms)
    plot(sizes, reshape(meandata(1,i,:), size(sizes)), '--o');
    plot(sizes, reshape(meandatadampened(1,i,:), size(sizes)), '-*');
end

legend('pow3', 'pow3damp', 'tanh', 'tanhdamp', 'jade', 'jadedamp', ...
    'yeredor', 'yeredordamp', 'fpca', 'fpcadamp');

save('plotdata.mat', 'plotdata');
savefig('AllAlgs');
print('All Algs','-dpng');

toc
diary off
