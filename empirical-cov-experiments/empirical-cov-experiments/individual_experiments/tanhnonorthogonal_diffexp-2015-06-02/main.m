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
exponents = '{2.01, 2.34333, 2.67667, 3.01, 3.34333, 3.67667, 4.01, 4.34333, 4.67667, 5.01}';

seed = 45;
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
orthmethods = cellstr(['covariance'; 'centroid  '; ...
                        'oracle    '; 'identity  ']);

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, length(orthmethods), length(sizes));

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
    end
    
    % Don't orthogonalize or damp
    rng(s);
    result = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'false', ...
            'algorithm', algorithm, ...
            'orthmethod', orthmethods{j}, ...
            'damp', 'false', ...
            'exponents', exponents, ...
            'regenerate_samples', 'false', ...
            'seed', seed, ...
            'run', i);
    plotdata(i,length(orthmethods),:) = result(2,:);
end

meandata = mean(plotdata,1);

figure()
hold on;
for i=1:length(orthmethods)
    plot(sizes, reshape(meandata(1,i,:), size(sizes)), '--o');
end

save('plotdata.mat', 'plotdata');

legend('covariance','centroid','oracle','identity');
 
savefig('FastICA - tanh');
print('FastICA - tanh','-dpng');

toc
diary off
