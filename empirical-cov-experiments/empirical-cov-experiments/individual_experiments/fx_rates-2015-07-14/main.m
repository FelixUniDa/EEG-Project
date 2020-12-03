%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

addpath('../../');
addpath('../../third_party');

data = csvread('../../fx_data/fx_rates.csv', 0, 1);
data = data(:,1:end-1);
mid = 2500;
train = data(1:mid,:);
test = data(mid+1:end,:);


% Import fx data

load('icasig2');
load('A2');
icasig=icasig2;

% Try some smoothing
smoothsig = zeros(size(icasig));
for i=1:size(icasig,1)
    
    x = 1:size(icasig,2);
    y = icasig(i,:);
    [pp,p] = csaps(x,y);
    figure(1);
    title('Click in figure to quit');
    subplot(2,1,1);
    pl = plot(x,icasig(i,:),'HitTest','off');
   
    %Do smoothing until user clicks on figure
    while true
        p = p/10;
        pp = csaps(x,y,p);
        subplot(2,1,1);
        plot(x,icasig(i,:));
        subplot(2,1,2);
        fnplt(pp);
        pause(1);

        if gco ~= pl
            break;
        end
    end
    close(gcf);
    
    %Final value of smoothed signal
    smoothsig(i,:) = csaps(x,y,p,x);
end

% Stationary version - MKO does not do this, but should be loo ked at

if false

data2 = data - repmat(data(1,:), size(data,1),1);
d = data2(1:end,:) - [zeros(1,size(data2,2)); data2(1:end-1,:)];
plot(1:length(d), d(:,1))

end


%Find AutoRegression Model
predsig = zeros(size(icasig,1),size(data,1)-mid);
for i=1:size(icasig,1)
    p = 3;
    [coeffs,err] = aryule(smoothsig(i,:), p);
    

    row = smoothsig(i,:);
    for j = mid+1:size(data,1)
        nextValue = -coeffs(2:end) * row(end:-1:end-p+1)';
        row = [row nextValue];
    end
    
    predsig(i,:) = row(mid+1:end);
end


xpred = A2*predsig;

figure(1);
subplot(2,1,1);
plot(test);
subplot(2,1,2);
plot(xpred');
   