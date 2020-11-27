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
p = 5;  %number of mixture recordings
A = randn(p,d); %random mixing matrix
%% Generate a 4-by-n array of source signals (independent components)
S = demosig;
%% Generate a random p-by-n mixture array
Y = A*S;
%% Center the data
Y = bsxfun(@minus, Y, mean(Y,2));
%% Whiten the data
[E,D]     = eig(cov(Y',1)); % EVD of sample covariance
[Ds,ord]    = sort( diag(D),'descend') ; %% Sort by decreasing variance
E = E(:,ord(1:d));      % E contains d largest eigenvectors
lam = Ds(1:d);          % vector of d largest eigenvalues
whiteningMatrix = diag(1./sqrt(lam))*E(:,1:d)';
dewhiteningMatrix = E(:,1:d) *diag(sqrt(lam));
X = whiteningMatrix*Y;
%% Use the PowerICA algorithm to estimate the d*d demixing matrix W
W0 = orth(randn(d,d)); % random initial start
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


