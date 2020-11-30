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
exponents = '{6,6,6,6,6,6,6,6,2.1,2.1}';

seed = 42;
rng(seed);

algorithm = 'tanh';

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

diary('output.txt');
tic

%% ------------------------------ Experiments -----------------------------

% We have to pad the strings because matlab is awful...
orthmethods = cellstr(['covariance     '; 'centroidscaling'; ...
                        'oracle         '; 'identity       ']);

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, length(orthmethods), length(sizes));
% Set up empy arrays to hold the data
amaridata = zeros(numberofruns, length(orthmethods), length(sizes));

% Run the comparison algorithm to get the data, storing it each time
for i = 1:numberofruns
    disp(['------------- Starting run ' int2str(i) ' ------------------']);
    
    s = rng;
    for j = 1:length(orthmethods)-1
        if j == 1
            regerenate_samples = 'true';
        else 
            regerenate_samples = 'false';
        end
        rng(s);
        result = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'false', ...
            'algorithm', algorithm, ...
            'orthmethod', orthmethods{j}, ...
            'damp', 'true', ...
            'regenerate_samples', regerenate_samples, ...
            'exponents', exponents, ...
            'seed', seed, ...
            'run', i);
        plotdata(i,j,:) = result(2,:);
        amaridata(i,j,:) = result(1,:);
    end
    
    % Don't orthogonalize or damp
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'false', ...
            'algorithm', algorithm, ...
            'damp', 'false', ...
            'exponents', exponents, ...
            'regenerate_samples', 'false', ...
            'seed', seed, ...
            'run', i);
    plotdata(i,length(orthmethods),:) = result(2,:);
    amaridata(i,length(orthmethods),:) = result(1,:);
end

save('amaridata.mat','amaridata');
save('plotdata.mat', 'plotdata');

%% Plot

load('plotdata.mat');

meandata = mean(plotdata,1);

figure();hold on;
set(gca,'DefaultTextFontSize',14);
set(gca,'FontSize',14);

plot(sizes, reshape(meandata(1,1,:), size(sizes)), '-or');
plot(sizes, reshape(meandata(1,2,:), size(sizes)), '-db');
plot(sizes, reshape(meandata(1,3,:), size(sizes)), '-*g');
plot(sizes, reshape(meandata(1,4,:), size(sizes)), '-+k');

y = ylim;
ylim([0 y(2)]);

legend('covariance','centroid','oracle','identity');
title('FastICA - tanh');
ylabel('Frobenius Error');
xlabel('Sample Size');
 
savefig('FastICA - tanh');
print('FastICA - tanh','-dpng');

load('amaridata.mat');

meandata = mean(amaridata,1);

figure();hold on;
set(gca,'DefaultTextFontSize',14);
set(gca,'FontSize',14);

plot(sizes, reshape(meandata(1,1,:), size(sizes)), '-or');
plot(sizes, reshape(meandata(1,2,:), size(sizes)), '-db');
plot(sizes, reshape(meandata(1,3,:), size(sizes)), '-*g');
plot(sizes, reshape(meandata(1,4,:), size(sizes)), '-+k');

y = ylim;
ylim([0 y(2)]);

legend('covariance','centroid','oracle','identity');
title('FastICA - tanh');
ylabel('Amari Index');
xlabel('Sample Size');
 
savefig('FastICA - tanh - amari');
print('FastICA - tanh -amari','-dpng');

toc
diary off
