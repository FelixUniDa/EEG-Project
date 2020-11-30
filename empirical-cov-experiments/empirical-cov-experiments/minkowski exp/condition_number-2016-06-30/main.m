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
lowerlimit = 1000;
upperlimit = 11000;
step = 2000;
exponents = '{6,6,6,6,6,6,6,6,2.1,2.1}';

seed = 422;
rng(seed);

algorithm = 'pow3';

numberofruns = 10;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');
tic

%% ------------------------------ Experiments -----------------------------

% We have to pad the strings because matlab is awful...
orthmethods = cellstr(['covariance     '; 'centroidscaling']);

% Set up empy arrays to hold the data
plotdata = zeros(numberofruns, length(orthmethods), length(sizes));

% Run the comparison algorithm to get the data, storing it each time
for i = 1:numberofruns
    disp(['--------- Starting run ' int2str(i) ' --------------']);

    s = rng;
    for j = 1:length(orthmethods)
        if j == 1
            regerenate_samples = 'true';
        else 
            regerenate_samples = 'false';
        end
        rng(s);
        [result, orthdata] = singlecomparison(n, lowerlimit, upperlimit, step, ...
            'verbose', 'true', ...
            'orthogonalmix', 'false', ...
            'algorithm', algorithm, ...
            'orthmethod', orthmethods{j}, ...
            'damp', 'true', ...
            'regenerate_samples', regerenate_samples, ...
            'exponents', exponents, ...
            'seed', seed, ...
            'run', i, ...
            'only', 'orthogonalize');
        plotdata(i,j,:) = orthdata;
    end
    
    % Don't orthogonalize or damp
%     rng(s);
%     [result, orthdata] = singlecomparison(n, lowerlimit, upperlimit, step, ...
%             'verbose', 'true', ...
%             'orthogonalmix', 'false', ...
%             'algorithm', algorithm, ...
%             'orthmethod', 'idenity', ...
%             'damp', 'false', ...
%             'exponents', exponents, ...
%             'regenerate_samples', 'false', ...
%             'seed', seed, ...
%             'run', i, ...
%             'only', 'orthogonalize');
%     plotdata(i,length(orthmethods),:) = orthdata;
end

meandata = mean(plotdata,1);

figure()
hold on;
for i=1:length(orthmethods)
    plot(sizes, reshape(meandata(1,i,:), size(sizes)), '-o');
end

title('Orthogonalization')
xlabel('Sample Size')
ylabel('Condition Number')

legend('covariance','centroid');
 
savefig('Condition_number');
print('Condition_number','-dpng');

filename = 'data_cond.mat';
save(filename,'-v4','meandata');

toc
diary off
