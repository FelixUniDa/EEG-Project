%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

addpath('../../');
addpath('../../third_party');

% Import stock data

[~,list] = system('ls -1 ../../stock_data');
files = textscan(list, '%s', 'delimiter', '\n');

rep = csvread(['../../stock_data/' files{1}{1}], 1, 1);

master_data = zeros(size(rep,1), length(files{1}));

for i=1:length(files{1})
    % disp(files{1}{i})
    tmp = csvread(['../../stock_data/' files{1}{i}], 1, 1);
    master_data(:,i) = padarray(tmp(:,4)-tmp(:,1), size(master_data,1)-size(tmp,1) , 0, 'post');
end

% [icasig, A, W] = fastica(master_data', 'displayMode', 'signals');
[icasig2, A2, W2] = fastica(damp(master_data',4), 'displayMode', 'signals');