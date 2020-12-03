% Import fx data

data = csvread('../../fx_data/fx_rates.csv', 0, 1);
data = data(:,1:end-1);

% Whiten the data
% white_data = data - repmat(mean(data,1),size(data,1),1);
% white_data = white_data./repmat(mean(white_data.^2).^(1/2),size(white_data,1),1);

% mid = floor(size(white_data,1)/2);
% train = white_data(1:mid,:);
% test = white_data(mid+1:end,:);

% mid = floor(size(data,1)/2);
mid = 2500;
train = data(1:mid,:);
test = data(mid+1:end,:);

% Do training stuff to fine tune:
% - damping parameter
% - smoothing tolerance
% - AR model

% Just to look at, the 7th column is fairly interesting
% figure(1);
% subplot(2,1,1);
% plot(1:size(train,1),train(:,7));

[icasig, A, W] = fastica(train', 'numOfIC', 4);
% [icasig2, A2, W2] = fastica(damp(rain',4), 'displayMode', 'signals');

save('icasig', 'icasig');
save('A','A');