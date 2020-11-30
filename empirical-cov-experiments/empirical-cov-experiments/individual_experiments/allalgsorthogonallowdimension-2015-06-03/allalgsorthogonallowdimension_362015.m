%%
% This is the main file to run the tests. If you don't want to run all
% the different algorithms, set the options you want in this section, run
% it, then run the section corresponding to the algorithm you care about.
% Each section of this code will call singlecomparison() to get plot data
% for the algorithm of choice. Refer to singlecomparison.m for more
% details.

n = 3;
lowerlimit = 1000;
upperlimit = 11000;
step = 2000;
exponents = '{6,6,2.1}';

seed = 44;
rng(seed);

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');
tic

%% ------------------------------ Experiments -----------------------------

algorithms = {'pow3','tanh','jade','yeredor'};

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, length(algorithms), length(sizes));
plotdatadamped = zeros(numberofruns, length(algorithms), length(sizes));

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
        plotdatadamped(i,j,:) = result(2,:);
    end
end


save('plotdata.mat', 'plotdata');
save('plotdatadamped.mat', 'plotdatadamped');

%% Plot

load('plotdata.mat');
load('plotdatadamped');

meandata = mean(plotdata,1);
meandatadamped = mean(plotdatadamped,1);

figure();hold on;
set(gca,'DefaultTextFontSize',14);
set(gca,'FontSize',14);

plot(sizes, reshape(meandata(1,1,:), size(sizes)), '-.or');
plot(sizes, reshape(meandatadamped(1,1,:), size(sizes)), '-or');
plot(sizes, reshape(meandata(1,2,:), size(sizes)), '-.*b');
plot(sizes, reshape(meandatadamped(1,2,:), size(sizes)), '-*b');
plot(sizes, reshape(meandata(1,3,:), size(sizes)), '-.xg');
plot(sizes, reshape(meandatadamped(1,3,:), size(sizes)), '-xg');

y = ylim;
ylim([0 y(2)]);

legend('FastICA - pow3', 'astICA - pow3 damped', 'FastICA - tanh', 'FastICA - tanh damped', 'JADE', 'JADE damped');
title('Damped vs. Raw Performance');
ylabel('Frobenius Error');
xlabel('Sample Size');

savefig('AllAlgs');
print('All Algs','-dpng');

toc
diary off
