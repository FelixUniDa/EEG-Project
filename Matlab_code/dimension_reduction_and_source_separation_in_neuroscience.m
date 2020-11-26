%%
%     COURSE: Dimension reduction and source separation in neuroscience
%    SECTION: Independent components analysis
%      VIDEO: ICA, PCA, GED on simulated data
% Instructor: mikexcohen.com
%
%%

% generate data
x = [ 1*randn(1000,1) .05*randn(1000,1) ];

% rotation matrix
th = -pi/6;
R1 = [ cos(th) -sin(th); sin(th) cos(th) ];
th = -pi/3;
R2 = [ cos(th) -sin(th); sin(th) cos(th) ];

% rotate the data to impose correlations
y = [ x*R1 ; x*R2 ];


% plot the data in its original data space
figure(1), clf
subplot(131), hold on
h(1) = plot(y(:,1),y(:,2),'o');
set(h(1),'color',[.8 .6 1])

% make the plot look nicer
datarange = max(y(:))*1.2;
set(gca,'xlim',[-datarange datarange],'ylim',[-datarange datarange])
xlabel('X axis'), ylabel('Y axis')
axis square
title('Data in XY space')


%%% PCA on data
y = bsxfun(@minus,y,mean(y,1));
covmat = (y'*y) / length(y);
[evecsY,evalsY] = eig(covmat);

% compute PC scores
pc1 = y*evecsY(:,1);
pc2 = y*evecsY(:,2);

% plot eigenvectors scaled by their eigenvalues
h(2:3) = plot([[0; 0] evecsY(1,:)']',[[0; 0] evecsY(2,:)']','r','linew',4);

% plot the data in PC space
subplot(132)
plot(pc2,pc1,'ms')
% make the plot look a bit nicer
datarange = max([pc1(:); pc2(:)])*1.2;
set(gca,'xlim',[-datarange datarange],'ylim',[-datarange datarange])
xlabel('PC1 axis'), ylabel('PC2 axis'), axis square
title('Data in PC space')




%%% run ICA
ivecs = jader(y');

% compute IC scores
ic_scores = ivecs*y';


% plot the IC vectors
subplot(131)
h(4:5) = plot([[0; 0] ivecs(1,:)']',[[0; 0] ivecs(2,:)']','k','linew',4);
legend(h([1 2 4]),{'Data';'PC';'IC'})

% plot the data in IC space
subplot(133)
plot(ic_scores(1,:),ic_scores(2,:),'ms')
datarange = max(ic_scores(:))*1.2;
set(gca,'xlim',[-datarange datarange],'ylim',[-datarange datarange])
xlabel('IC1 axis'), ylabel('IC2 axis'), axis square
title('Data in IC space')

%%





%% part 2: simulated EEG data

% a clear MATLAB workspace is a clear mental workspace
close all; clear all

%%

% load mat file containing EEG, leadfield and channel locations
load emptyEEG

% pick a dipole location in the brain
diploc = 109;


% normalize dipoles
lf.GainN = bsxfun(@times,squeeze(lf.Gain(:,1,:)),lf.GridOrient(:,1)') + bsxfun(@times,squeeze(lf.Gain(:,2,:)),lf.GridOrient(:,2)') + bsxfun(@times,squeeze(lf.Gain(:,3,:)),lf.GridOrient(:,3)');


% plot brain dipoles
figure(1), clf, subplot(221)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'o')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 's','markerfacecolor','w','markersize',10)
rotate3d on, axis square, axis off
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(222)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Signal dipole projection')


% Now we generate random data in brain dipoles.
% create 1000 time points of random data in brain dipoles
% (note: the '1' before randn controls the amount of noise)
dipole_data = 1*randn(length(lf.Gain),1000);

% add signal to second half of dataset
dipole_data(diploc,501:end) = 15*sin(2*pi*10*(0:499)/EEG.srate);

% project dipole data to scalp electrodes
EEG.data = lf.GainN*dipole_data;

% meaningless time series
EEG.times = (0:size(EEG.data,2)-1)/EEG.srate;

% plot the data from one channel
subplot(212), hold on, plot(.5,0,'HandleVisibility','off');
plot(EEG.times,dipole_data(diploc,:)/norm(dipole_data(diploc,:)),'linew',4)
plot(EEG.times,EEG.data(31,:)/norm(EEG.data(31,:)),'linew',2)
xlabel('Time (s)'), ylabel('Amplitude (norm.)')
legend({'Dipole';'Electrode'})

%% Create covariance matrices

% compute covariance matrix R is first half of data
tmpd = EEG.data(:,1:500);
tmpd = bsxfun(@minus,tmpd,mean(tmpd,2));
covR = tmpd*tmpd'/500;

% compute covariance matrix S is second half of data
tmpd = EEG.data(:,501:end);
tmpd = bsxfun(@minus,tmpd,mean(tmpd,2));
covS = tmpd*tmpd'/500;


%%% plot the two covariance matrices
figure(2), clf

% S matrix
subplot(131)
imagesc(covS)
title('S matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R matrix
subplot(132)
imagesc(covR)
title('R matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R^{-1}S
subplot(133)
imagesc(inv(covR)*covS)
title('R^-^1S matrix')
axis square, set(gca,'clim',[-10 10])


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%                                 %
%  Dimension compression via PCA  %
%                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PCA
[evecs,evals] = eig(covS+covR);

% sort eigenvalues/vectors
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);



% plot the eigenspectrum
figure(3), clf
subplot(231)
plot(evals./max(evals),'s-','markersize',15,'markerfacecolor','k')
axis square
set(gca,'xlim',[0 20.5])
title('PCA eigenvalues')
xlabel('Component number'), ylabel('Power ratio (norm-\lambda)')


% component time series is eigenvector as spatial filter for data
comp_ts = evecs(:,1)'*EEG.data;


% normalize time series (for visualization)
dipl_ts = dipole_data(diploc,:) / norm(dipole_data(diploc,:));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = EEG.data(31,:) / norm(EEG.data(31,:));


% plot the time series
subplot(212), hold on
plot(EEG.times,.3+dipl_ts,'linew',2)
plot(EEG.times,.15+chan_ts)
plot(EEG.times,comp_ts)
legend({'Truth';'EEG channel';'PCA time series'})
set(gca,'ytick',[])
xlabel('Time (a.u.)')


%% spatial filter forward model

% The filter forward model is what the source "sees" when it looks through the
% electrodes. It is obtained by passing the covariance matrix through the filter.
filt_topo = evecs(:,1);

% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection electrode to be positive.
[~,se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));


% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Truth topomap')

subplot(233)
topoplotIndie(filt_topo,EEG.chanlocs,'electrodes','numbers','numcontour',0);
title('PCA forward model')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%                                 %
%    Source separation via GED    %
%                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generalized eigendecomposition (GED)
[evecs,evals] = eig(covS,covR);

% sort eigenvalues/vectors
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);



% plot the eigenspectrum
figure(4), clf
subplot(231)
plot(evals./max(evals),'s-','markersize',15,'markerfacecolor','k')
axis square
set(gca,'xlim',[0 20.5])
title('GED eigenvalues')
xlabel('Component number'), ylabel('Power ratio (norm-\lambda)')

% component time series is eigenvector as spatial filter for data
comp_ts = evecs(:,1)'*EEG.data;

%% plot for comparison

% normalize time series (for visualization)
dipl_ts = dipole_data(diploc,:) / norm(dipole_data(diploc,:));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = EEG.data(31,:) / norm(EEG.data(31,:));


% plot the time series
subplot(212), hold on
plot(EEG.times,.3+dipl_ts,'linew',2)
plot(EEG.times,.15+chan_ts)
plot(EEG.times,comp_ts)
legend({'Truth';'EEG channel';'GED time series'})
set(gca,'ytick',[])
xlabel('Time (a.u.)')


%% spatial filter forward model

% The filter forward model is what the source "sees" when it looks through the
% electrodes. It is obtained by passing the covariance matrix through the filter.
filt_topo = covS*evecs(:,1);

% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection electrode to be positive.
[~,se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));


% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Truth topomap')

subplot(233)
topoplotIndie(filt_topo,EEG.chanlocs,'electrodes','numbers','numcontour',0);
title('GED forward model')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%                                 %
%    Source separation via ICA    %
%                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% run ICA and compute IC time series and maps
icvecs = jader(EEG.data,20);
ICs    = icvecs(1,:)*EEG.data;
icmaps = pinv(icvecs');
ICenergy = sum(icmaps.^2,2);

figure(5), clf

% plot component energy
subplot(231)
plot(ICenergy./max(ICenergy),'s-','markersize',15,'markerfacecolor','k')
axis square
set(gca,'xlim',[0 20.5])
title('IC energy')
xlabel('Component number'), ylabel('Power ratio (norm-\lambda)')



% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Truth topomap')

subplot(233)
topoplotIndie(icmaps(1,:),EEG.chanlocs,'electrodes','numbers','numcontour',0);
title('IC forward model')


% plot the time series
subplot(212), hold on
plot(EEG.times,.3+dipl_ts,'linew',2)
plot(EEG.times,.15+chan_ts)
plot(EEG.times,ICs)
legend({'Truth';'EEG channel';'IC time series'})
set(gca,'ytick',[])
xlabel('Time (a.u.)')

%% done.