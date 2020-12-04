%% Set up parameters
addpath('../');
addpath('../third_party');

clear variables;

% verbose = true;
% orthogonalmix = false;
% maxTries1 = 100; % Number of times to retry plain ICA if it fails
% maxTries2 = 10; % Number of times to retry HTICA if it fails

% numberOfRuns = 10;

% Sub-folders of data/
data_folders = {
    'cocktail_022309_ceilhex'%, ...
    %'cocktail_042409_perimeterreg'%, ...
    %'cocktail_020909_ceilreg', ...
    %'cocktail_040909_ceilrand', ...
    %'cocktail_051409_endfire'
};
 
% seeds = [10 43 42 45];
% seeds = [100 60];

main_verbose = true;
main_orthogonalmix = false;
main_maxTries1 = 100; % Number of times to retry plain ICA if it fails
main_maxTries2 = 10; % Number of times to retry HTICA if it fails

main_numberOfRuns = 10;

% seeds = [10 43 42 45];
main_seeds = [100];

main_start = 100000;
main_sizes = (1:5)*10000;

main_samplingRate = 2;

numberOfFolders = length(data_folders);

tic

for folder_i = 1:numberOfFolders
    par_data_folders = data_folders;
    data_folder = par_data_folders{folder_i};
    
    % "main_" prefixed variables are to accommodate parfor transparency
    verbose = main_verbose;
    orthogonalmix = main_orthogonalmix;
    maxTries1 = main_maxTries1; % Number of times to retry plain ICA if it fails
    maxTries2 = main_maxTries2; % Number of times to retry HTICA if it fails

    numberOfRuns = main_numberOfRuns;

    seeds = main_seeds;
    
    start = main_start;
    sizes = main_sizes;
    
    samplingRate = main_samplingRate;
    
    for seed_i = 1:length(seeds)
        seed = seeds(seed_i);
        rng(seed);

        disp(['========= Executing experiment in data folder ' ...
            data_folder ' with seed ' num2str(seed) '...']);

        %% Diary

        delete(['data/' data_folder '/output.txt']);
        diary(['data/' data_folder '/output.txt']);

        %% Read Party Data
        [S,Fs] = audioread(['data/' data_folder '/party1.wav']);
        S = S';
        [n, m] = size(S);
        
        %% Read Individual Signals
        
        trueSignals = zeros(6,size(S,2));
        for i = 1:6
            fname = ['data/' data_folder '/soi' num2str(i) '.wav'];
            [y,Fs] = audioread(fname);
            trueSignals(i,:) = y(:, max(y,[],1) == max(max(y,[],1)));
        end
        
        %% Separate Signals
        
        for size_i = 1:length(sizes)
            [~, Aest, West] = fastica(S(:,1:sizes(size_i)), ...
                'g', 'pow3', ...
                'verbose', 'off', ...
                'numOfIC', 5 ...
            );
            
            figure();
            strips((West*S)');
            title(['Estimated signals using ' num2str(sizes(size_i)) ' samples']);
        end

        %% Plot data. Can be run independently
        
        figure();
        strips(trueSignals');
        title('True Signals');
        figure();
        strips(S');
        title('Mixed Signals');

        if false

        h = figure();
        set(h, 'Position', [1000 400 800 800]);
        hold on;

        % Plot ICA - pow3 error on full data
        plot();

        title(['Error of ' data_folder]);
        ylabel('Frobenius Error');
        xlabel('Sample Size');
        legend('ICA - pow3 Baseline', ...
            'ICA - tanh Baseline', ...
            'ICA - pow3', ...
            'HTICA', ...
            'ICA - tanh');
        
        set(gca,'DefaultTextFontSize',16);
        set(gca,'FontSize',16);
        y = ylim;
        %ylim([0 y(2)]);
        axis square;

        fname = ['data/' data_folder '/error-seed-' num2str(seed) '-start-' num2str(start) '-samplingRate-' num2str(samplingRate)];
        savefig(fname);
        print(fname,'-dpng');
        
        end

        %%

    end
end

toc
diary off