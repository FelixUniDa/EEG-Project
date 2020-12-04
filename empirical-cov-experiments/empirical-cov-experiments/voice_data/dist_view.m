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
    %'aalto_data' %, ... % Six SOI and 5000 samples each
    %'cocktail_022309_ceilhex'%  , ... % Six SOI, 551250 samples
    %'cocktail_042409_perimeterreg' %, ... % Six SOI, 551250 samples
    'cocktail_020909_ceilreg'%, ... % Six SOI, 551250 samples
    %'cocktail_040909_ceilrand', ... % Six SOI, 551250 samples
    %'cocktail_051409_endfire' % Six SOI, 551250 samples
};

numberOfFolders = length(data_folders);

main_verbose = true;
main_shuffle = false;
main_start = 1;
main_samplingRate = 1;

warning ('off','all');

for folder_i = 1:numberOfFolders
    par_data_folders = data_folders;
    data_folder = par_data_folders{folder_i};
    disp(['Using data in ' data_folder '...']);
    
    verbose = main_verbose;
    shuffle = main_shuffle;
    start = main_start;
    samplingRate = main_samplingRate;
    
    %% ================== READ DATA INTO S ========================
        [y,Fs] = audioread(['data/' data_folder '/soi1.wav']);
        S = zeros(size(y,1), 6);
        for i = 1:6
            fname = ['data/' data_folder '/soi' num2str(i) '.wav'];
            [y,Fs] = audioread(fname);
            S(:,i) = y(:, max(y,[],1) == max(max(y,[],1)));
        end
        S = S';
        % S = S(:, 1:4000);
        [n, m] = size(S);
        
        % Center the samples
        S = S - repmat(mean(S,2), 1, m);
        
        % Symmetrize
        if mod(m,2) == 1
            S = S(:,1:end-1);
            m = m - 1;
        end
        S = S(:,1:m/2) - S(:,(m/2)+1:end);
        
        alphas = alphaest(S); % n-by-1 alphas
        gammas = gammaest(S); % n-by-1 gammas
        
        for component_i = 1:length(alphas)
            if false
                pd1 = makedist('Stable','alpha',alphas(component_i), ...
                    'beta',0,'gam',gammas(component_i),'delta',0);
                [f,xi] = ksdensity(S(component_i,:)); % S density at xi's

                pdf1 = pdf(pd1,xi);

                figure();
                plot(xi, pdf1, 'b-');
                hold on;
                plot(xi, f, 'r-');
            end
            figure();
            histfit(S(component_i,:), 100, 'stable');
            disp(['Component ' num2str(component_i) ' fitdist params:']);
            fitdist(S(component_i,:)', 'stable')
            figure();
            histfit(S(component_i,:), 100, 'normal');
        end
end