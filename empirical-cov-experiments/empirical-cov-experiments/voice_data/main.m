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
    %'cocktail_022309_ceilhex'  , ... % Six SOI, 551250 samples
    'cocktail_042409_perimeterreg', ... % Six SOI, 551250 samples
    %'cocktail_020909_ceilreg'%, ... % Six SOI, 551250 samples
    %'cocktail_040909_ceilrand', ... % Six SOI, 551250 samples
    %'cocktail_051409_endfire' % Six SOI, 551250 samples
};
 
% seeds = [10 43 42 45];
% seeds = [100 60];

main_verbose = true;
main_orthogonalmix = false;
main_maxTries1 = 100; % Number of times to retry plain ICA if it fails
main_maxTries2 = 10; % Number of times to retry HTICA if it fails

main_numberOfRuns = 10;

main_seeds = [100 42];

main_start = 200000;
main_sizes = (1:10)*1000;

main_shuffle = false;

main_samplingRate = 1;

% main_R = 24; % Accepts about 90% of the samples
main_R = 14; % Accepts about 75% of the samples

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
    
    shuffle = main_shuffle;
    
    samplingRate = main_samplingRate;
    
    R = main_R;
    
    for seed_i = 1:length(seeds)
        seed = seeds(seed_i);
        rng(seed);

        disp(['========= Executing experiment in data folder ' ...
            data_folder ' with seed ' num2str(seed) '...']);

        %% ================================================================

        delete(['data/' data_folder '/output.txt']);
        diary(['data/' data_folder '/output.txt']);

        %% ================================================================
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

        % Generate random mixing matrix A from standard gaussian
        A = mvnrnd(zeros(1,n), eye(n), n);
        % Normalize columns of A
        A = A*(inv(diag(rownorm(A'))));
        if orthogonalmix
            A = orth(A); % Optional, based on whether we want an orthonormal basis
            % A = eye(n);
        end
        % disp('Mixing matrix:')
        % disp(A)

        X = A*S;
        % X = X(:, 1:samplingRate:end);
        X = resample(X', 1, samplingRate);
        X = X';
        % size(X)
        
        % Checking to see if we have enough samples for the experiments
        if size(X,2) < start + sizes(end)
            error(['Asking for ' num2str(start + sizes(end)) ' samples and only ' num2str(size(X,2)) ' available.']);
        end

        % Use ICA on the full dataset and save for later
        [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n);
        Aest = [Aest zeros(6,n-size(Aest,2))];
        scale = diag(rownorm(Aest')).^(-1);
        scale(scale == Inf) = 0;
        Aest = Aest * scale;
        [~, AestCorrected] = basisEvaluation(A, Aest);
        fullPow3DataError = norm(A-AestCorrected, 'fro');

        % clear AestFull AestFullCorrected
        AestFull = [];
        AestFullCorrected = [];

        % Use ICA on the full dataset and save for later
        [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n, 'g', 'tanh');
        Aest = [Aest zeros(6,n-size(Aest,2))];
        scale = diag(rownorm(Aest')).^(-1);
        scale(scale == Inf) = 0;
        Aest = Aest * scale;
        [~, AestCorrected] = basisEvaluation(A, Aest);
        fullTanhDataError = norm(A-AestCorrected, 'fro');

        numberOfSizes = length(sizes);

        % First row: plain fastica using pow3
        % Second row: htica
        % Third row: fastica with tanh
        errors = zeros(3, numberOfSizes, numberOfRuns);

        for i = 1:numberOfSizes
            for j = 1:numberOfRuns
                disp(['==== STARTING RUN ' num2str(j) ' with ' ...
                    num2str(sizes(i)) ' samples ====']);

                % clear Xslice
                Xslice = [];

                % ===== FastICA - pow3 =====

                % clear Aest Aestcorrected
                Aest = [];
                AestCorrected = [];
                tries = 0;

                % Xslice = X(:,(start+1):(start + i*1000));
                
                if shuffle
                    % Select a random set of the points
                    inds = randperm(size(X,2), sizes(i));
                    Xslice = X(:, inds);
                    
                    %disp(['Using indexes ' num2str(inds)]);
                    disp(['Using ' num2str(sizes(i)) ' shuffled samples']);
                else
                    Xslice = X(:, start:(start+sizes(i)));
                
                    disp(['Using indexes ' num2str(1) ' to ' ...
                        num2str(sizes(i))]);
                end

                [~, Aest, ~] = fastica(Xslice, 'numOfIC', n, ...
                                        'verbose', 'off');

                while size(Aest,2) < n
                    tries = tries + 1;
                    if tries > maxTries1
                        if verbose
                            disp('Plain ICA failed...')
                        end
                        Aest = [Aest zeros(6,n-size(Aest,2))];
                    else
                        [~, Aest, ~] = fastica(Xslice, ...
                                        'verbose', 'off', ...
                                        'numOfIC', n);
                    end
                end

                scale = diag(rownorm(Aest')).^(-1);
                scale(scale == Inf) = 0;
                Aest = Aest * scale;
                [~, Aestcorrected] = basisEvaluation(A,Aest);

                error = norm(A-Aestcorrected,'fro');
                errors(1, i, j) = error;
                disp(['FastICA - pow3 error: ' num2str(error)]);

                % ===== HTICA =====

                % clear Aest Aestcorrected
                Aest = [];
                AestCorrected = [];
                % clear orthogonalizer XsliceOrth
                orthogonalizer = [];
                XsliceOrth = [];
                tries = 0;

                disp('Orthogonalizing Data...');
                orthogonalizer = centroidOrthogonalizer(Xslice, 'scale');
                % C = (1/m) * (X * X');
                % orthogonalizer = inv(sqrtm(C));
                XsliceOrth = orthogonalizer * Xslice;

                % R = 24;
                R = main_R;

                % disp('Damping tails...');
                [Xslice, rate, ~] = damp(XsliceOrth, R);
                disp(['Acceptance rate: ' num2str(rate)]);
                [~, Aest, ~] = fastica(Xslice, 'verbose', 'off', ...
                                'numOfIC', 6);
                failed = false;
                while size(Aest,2) < n
                    tries = tries + 1;
                    if tries > maxTries2
                        if R > 0
                            disp(['Accepting more samples with R = ' ...
                                num2str(R)]);
                            R = R - 1;
                            tries = 0;
                        else
                            if verbose
                                disp('HTICA failed. Padding with zero...')
                            end
                            Aest = [Aest zeros(n,n-size(Aest,2))];
                            Aest(isnan(Aest)) = 0;
                            failed = true;
                        end
                    else
                        Aest2 = [];
                        [Xslice, rate, ~] = damp(XsliceOrth, R);
                        [~, Aest, ~] = fastica(Xslice, ...
                                        'verbose', 'off', ...
                                        'numOfIC', n);
                    end
                end

                if ~failed
                    Aest = inv(orthogonalizer) * Aest;
                end

                scale = diag(rownorm(Aest')).^(-1);
                scale(scale == Inf) = 0;
                Aest = Aest * scale;
                [~, Aestcorrected] = basisEvaluation(A,Aest);

                error = norm(A-Aestcorrected,'fro');
                errors(2, i, j) = error;
                disp(['HTICA error: ' num2str(error)]);

                % ===== FastICA - tanh =====

                % clear Aest Aestcorrected
                Aest = [];
                AestCorrected = [];
                tries = 0;

                [~, Aest, ~] = fastica(Xslice, 'numOfIC', n, ...
                                'verbose', 'off', ...
                                'g', 'tanh');

                while size(Aest,2) < n
                    tries = tries + 1;
                    if tries > maxTries1
                        if verbose
                            disp('Plain ICA failed...')
                        end
                        Aest = [Aest zeros(6,n-size(Aest,2))];
                    else
                        [~, Aest, ~] = fastica(Xslice, ...
                                        'verbose', 'off', ...
                                        'g', 'tanh', 'numOfIC', n);
                    end
                end

                scale = diag(rownorm(Aest')).^(-1);
                scale(scale == Inf) = 0;
                Aest = Aest * scale;
                [~, Aestcorrected] = basisEvaluation(A,Aest);

                error = norm(A-Aestcorrected,'fro');
                errors(3, i, j) = error;
                disp(['FastICA - tanh error: ' num2str(error)]);
            end
        end

        avgErrors = mean(errors,3);

        % save(['data/' data_folder '/errors.mat'], 'errors');
        % save(['data/' data_folder '/avgErrors.mat'], 'avgErrors');
        % save(['data/' data_folder '/fullPow3DataError.mat'], ...
        %     'fullPow3DataError');
        % save(['data/' data_folder '/fullTanhDataError.mat'], ...
        %     'fullTanhDataError');
        % save(['data/' data_folder '/seeds.mat'], ...
        %     'seeds');

        %% Plot data. Can be run independently

        % load(['data/' data_folder '/errors.mat']);
        % load(['data/' data_folder '/avgErrors.mat']);
        % load(['data/' data_folder '/fullPow3DataError.mat']);
        % load(['data/' data_folder '/fullTanhDataError.mat']);
        % load(['data/' data_folder '/seeds.mat']);

        h = figure();
        set(h, 'Position', [1000 400 800 800]);
        hold on;

        % Plot ICA - pow3 error on full data
        plot(sizes, repmat(fullPow3DataError, 1, numberOfSizes));

        % Plot ICA - tanh error on full data
        plot(sizes, repmat(fullTanhDataError, 1, numberOfSizes));

        % Plot plain ICA error
        plot(sizes, avgErrors(1,:), '-o')

        % Plot HTICA error
        plot(sizes, avgErrors(2,:), '-o')

        % Plot FastICA with tanh error
        plot(sizes, avgErrors(3,:), '-o')

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

        if shuffle
            fname = ['data/' data_folder '/error-seed-' num2str(seed) '-shuffled-samplingRate-' num2str(samplingRate) '-R-' num2str(R)];
        else
            fname = ['data/' data_folder '/error-seed-' num2str(seed) '-start-' num2str(start) '-samplingRate-' num2str(samplingRate) '-R-' num2str(R)];
        end
        savefig(fname);
        print(fname,'-dpng');

        %%

    end
end

toc
diary off