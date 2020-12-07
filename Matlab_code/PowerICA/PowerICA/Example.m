%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                               Example                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a synthetic example of utilizing the PowerICA method in
% extracting independent source signals from their observed mixture 
% recordings.
%
% Code by Shahab Basiri, Aalto University 2017 (shahab.basiri@aalto.fi).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
d = 4;  %number of ICs
p = 4;  %number of mixture recordings -> normally 5
A = randn(p,d); %random mixing matrix
%% Generate a 4-by-n array of source signals (independent components)
S1 = demosig;
plot(S1(2,:))
times = 0:0.05:100-0.05;
s1 = sin(pi*times);
s2 = sin(pi*times*0.5);
s3 = cos(pi*times*4);
s4 = cos(pi*times*2);
S = [s1; s2; s3; s4];
%% Generate a random p-by-n mixture array
%A = [0.4766    0.5000    0.1724    0.1999;
%    0.4900   -0.7696    0.4955    1.3449;
%    0.2058   -1.3669   -1.4839   -1.1978;
%   0.6802    0.4585   -0.6467    0.1311];
Y = A*S;
%% Center the data
Y = bsxfun(@minus, Y, mean(Y,2));
%% Whiten the data
[E,D]     = eig(cov(Y',1)); % EVD of sample covariance
[Ds,ord]    = sort( diag(D),'descend') ; %% Sort by decreasing variance
E = E(:,ord(1:d));      % E contains d largest eigenvectors
lam = Ds(1:d);          % vector of d largest eigenvalues
%D_inv = diag(1./sqrt(lam));
%E_transp = E(:,1:d)';
whiteningMatrix = diag(1./sqrt(lam))*E(:,1:d)';
dewhiteningMatrix = E(:,1:d) *diag(sqrt(lam));
X = whiteningMatrix*Y;
%% Use the PowerICA algorithm to estimate the d*d demixing matrix W
W0 = orth(randn(d,d)); % random initial start %W0 = [0.4995    0.3415    0.5264    0.5973;
%    0.5879   -0.7635    0.1708   -0.2057;
%   -0.5648   -0.2360    0.7861   -0.0855;
%    0.2930    0.4947    0.2753   -0.7704];
nonlin = 'tanh'; %ICA nonlinearity 
mode = 'serial'; % computation mode may be changed to 'parallel'
[W_est , flg] = PowerICA(X, nonlin, W0, mode); 
%% PowerICA estimate of the ICs
S_est = W_est*X;
%% PowerICA estimate of A (up to sign and permutation ambiguities)
A_est = dewhiteningMatrix*W_est';
fprintf('The powerICA estimate of A is\n');
display(A_est);
%% Plotting
% Source signals (independent components)
figure;
subplot(4,1,1);plot(S(1,:));ylabel('IC#1');
title('Source signals (independent components)');
subplot(4,1,2);plot(S(2,:));ylabel('IC#2');
subplot(4,1,3);plot(S(3,:));ylabel('IC#3');
subplot(4,1,4);plot(S(4,:));ylabel('IC#4');
% Mixtures
figure;
for i = 1:p
subplot(p,1,i);plot(Y(i,:));ylabel(['MIX#' num2str(i)]);
if i == 1, title('Observed mixtures');end
end
% PowerICA estimate of source signals (independent components)
% find the order of extraction:
W = (whiteningMatrix*A)';
[~, ord]=max(abs(W_est/W),[],2);
figure;
subplot(4,1,1);plot(S_est(1,:));ylabel(['IC#' num2str(ord(1))]);
title('PowerICA estimate of source signals');
subplot(4,1,2);plot(S_est(2,:));ylabel(['IC#' num2str(ord(2))])
subplot(4,1,3);plot(S_est(3,:));ylabel(['IC#' num2str(ord(3))])
subplot(4,1,4);plot(S_est(4,:));ylabel(['IC#' num2str(ord(4))]);


