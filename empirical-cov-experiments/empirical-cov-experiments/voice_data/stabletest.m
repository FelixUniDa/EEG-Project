addpath('../');
addpath('../third_party');

clear variables;

% Sub-folders of data/
data_folders = {
%     'aalto_data'
     'cocktail_022309_ceilhex', ...
     'cocktail_042409_perimeterreg', ...
     'cocktail_020909_ceilreg'%, ...
%     'cocktail_040909_ceilrand', ...
%     'cocktail_051409_endfire'
};

numberOfFolders = length(data_folders);

tic

for folder_i = 1:numberOfFolders
    par_data_folders = data_folders;
    data_folder = par_data_folders{folder_i};
    
    disp(['========= Executing experiment in data folder ' ...
            data_folder '...']);
    
    [y,Fs] = audioread(['data/' data_folder '/soi1.wav']);
    S = zeros(size(y,1), 6);
    for i = 1:6
        fname = ['data/' data_folder '/soi' num2str(i) '.wav'];
        [y,Fs] = audioread(fname);
        S(:,i) = y(:, max(y,[],1) == max(max(y,[],1)));
    end
    S = S';
    [n, m] = size(S);
    
    S(S == 0) = 0.0000001;
    
    % block_size = 500;
    
    step_size = 10000;
    start = 1;
    
    block_alphas = zeros(n, length(start:step_size:m));    
    window_chunks = start:step_size:m;
    
    for chunk_i = 1:length(window_chunks)
        slice = S(:,window_chunks(chunk_i):min(window_chunks(chunk_i)+step_size-1,m));
        block_alphas(:,chunk_i) = alphaest(slice);
    end
    
    start = 10000;
    front_chunks = start:step_size:m;
    front_alphas = zeros(n, length(front_chunks));
    for chunk_i = 1:length(front_chunks)
        slice = S(:,1:front_chunks(chunk_i));
        front_alphas(:,chunk_i) = alphaest(slice);
    end
    
    %% Plot Windows
    
    h = figure(folder_i);
    set(h, 'Position', [1000 400 800 800]);
    hold on;
    set(gca,'DefaultTextFontSize',16);
    set(gca,'FontSize',16);
    title(['Alpha Estimates of ' data_folder ' - Blocks of ' num2str(step_size)]);
    xlabel(['Window Starting Point (' num2str(step_size) ' samples in window)']);
    ylabel('Estimated alpha');
    
    for signal_i = 1:n
        plot(window_chunks, block_alphas(signal_i, :), '--o');
    end
    
    y = ylim;
    ylim([0 y(2)]);
    
    fname = ['data/' data_folder '/window-alphas'];
    savefig(fname);
    print(fname,'-dpng');
    
    %% Plot Front Blocks
    
    h = figure(numberOfFolders + folder_i);
    set(h, 'Position', [1000 400 800 800]);
    hold on;
    set(gca,'DefaultTextFontSize',16);
    set(gca,'FontSize',16);
    title(['Alpha Estimates of ' data_folder ' - Front Blocks of ' num2str(step_size)]);
    xlabel('Number of Samples');
    ylabel('Estimated alpha');
    
    for signal_i = 1:n
        plot(front_chunks, front_alphas(signal_i, :), '--o');
    end
    
    y = ylim;
    %ylim([0 y(2)]);
    axis square;
    
    fname = ['data/' data_folder '/front-alphas'];
    savefig(fname);
    print(fname,'-dpng');
end