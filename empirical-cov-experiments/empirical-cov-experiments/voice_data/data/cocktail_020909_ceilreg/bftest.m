%  This script provides an example of reading in experimental data
%  files associated with a series of cocktail recordings useful for the
%  test/development of beamforming and source separation algorithms.
%  Exeriments were recorded with spatially distributed microphones
%  where the conversation of interest (COI) was recorded separately from
%  the multiple simultaneous conversations (party).  The microphone
%  geometry and settings were the same of both recordings, so they can
%  be superimposed at specified power ratios and separated for
%  performance metric estimations, such as SNR and intelligibility.
%
%  This script opens the data and parameter files and in a hopping window
%  fashion, performs a delay and sum beaformer on the COI in party noise.
%  there are weights applied to each microphone channel based on thier
%  distances from the source.
%
%  Parameters for adjustment:
%  The balance between close and distance mics can be adjusted base on
%  parameter WP described in the script.
%  The SNR between the COI and party noise can also be set
%  with parameter SNRDB_DESIRED.
%  A high-pass filter is applied to remove long wavelengths resulting
%  from room modes with parameter HPF which is cutoff in Hz
%  The window size can be change with WS in seconds.
%   The speaker of interest file and the party noise can also be selected
%   by changing the wave file names in the script SOI#.wav and PARTY#.wav.
%
%   Outputs
%  The script creates a plot of source and microphone positions
%  so the geometry can be observed, it then plays and plots the
%  solo COI using the closest mic, the COI plus party noise using
%  the closest mic, and finally the beamformed COI.
%  Mean intelligibilty and SNR values before and after beamforming are
%  printed to the workspace, silence intervals are removed before computing
%  the mean intelligibility.
%
%     Written by Kevin D. Donohue (donohue@engr.uky.edu) Sept. 2008
%

% Processing paraemters
% wp => mic weight distibtion where 0 results in equal weights,
% a positive number gives more weight to closer mics, and
% a negative number gives more weight to distant mics:
wp = 1;
% Set SNR in dB for conversation of interest relative to 
%  cocktail party noise (average power ratio over all mics)
snrdb_desired = 2;
% Processing window size in seconds for reading in data and applying 
% the current source position number assoicated with the correspond
% time the window includes.
ws = 40e-3;
% High-pass cutoff in Hertz for filtering signal before beamforming
fc = 300;
% Name of multichannel wavefile with source recordings
sfile = 'soi2-orig.wav'; 
% Name of multichannel wavefile with cocktail party/noise recordings
nfile = 'party2.wav';


% Name of file with source positions listed over time
sposfile = [sfile(1:4) 'pos.txt']; 
% Name of file with static microphone position
mposfile = 'mpos.txt';
% Name of file with measured environment parameters (i.e. speed of sound) 
parmfile = 'info.txt';

% Load files associated with recordings
spos = load(sposfile,'-ascii');  %  Source Positions

mpos = load(mposfile,'-ascii'); % Mic Postions
%mpos = mpos/100;  %  Convert mic positions to meters
%  Get speed of sound out of file
fid = fopen(parmfile);            
h = textscan(fid,'%s');
for k=1:length(h{1})
    if strmatch(h{1}(k),'=')
        c = str2double(h{1}(k+1));
        break;
    end
end

fclose(fid);
%  Plot Mic and Speaker geometry
figure
plot3(mpos(1,:),mpos(2,:),mpos(3,:),'ob')
axis([0 4 0 4 0 2.7]);  % Set axis on the order of recording space
grid
hold on %  Superimpose speaker positions
plot3(spos(:,2),spos(:,3),spos(:,4),'xr')
xlabel('Meters X')
ylabel('Meters Y')
zlabel('Meters Z')
title('Mic Positions (Blue o), Source Positions (red x)')
hold off
pause(.2)

%  Dummy reads to get file information and compute
%  average power in signals and noise
[y, fs] = audioread(nfile);
[siglen1,chans] = size(y);
nospower = mean(std(y).^2);
[y, fs] = audioread(sfile);
[siglen2, chans] = size(y);
samps(1) = min([siglen1, siglen2]);
samps(2) = chans;
% Compute linear scaling factor to achieve SNR 
sigpower = mean(std(y).^2);
dBgain = snrdb_desired - 10*log10(sigpower/(nospower+eps));
swt = 10^(dBgain/20);
% Compute filter coefficients
[b,a] = butter(4,fc/(fs/2),'high');

%  Initalize sample indecies to step through
%  recordings one window at a time
segstart = 1;
segend = segstart+round(ws*fs);

%  If current segment end sample is less than total signal length
%   beamform the data
wsc = ws;
segstart =  1;
segend = segstart+round(wsc*fs);
loopcount = 0;
y = [];
totclosemic = [];sig_beam=[];nos_beam=[];
sigclosemic = [];
nosclosemic=[];
while segend <= samps(1)
    % Read in short segment of data form
    x= audioread(nfile,[segstart segend]);  % cocktail party
    xn = x; % cocktail party noise signal
    [xi,~] = audioread(sfile,[segstart segend]); % conversation of interest (COI)
    xi = swt*xi; % scaled converstion of interest(COI) at specfied SNR
    x = x+ xi;  %  Combine COI with party noise at specified average SNR
    
    %  If first time in loop initalize, otherwise update from previous
    if loopcount > 0;  % Update
       [x,xf1] = filter(b,a,x,xf1);
       [x_nos,xf1_nos] = filter(b,a,xn,xf1_nos);
       [x_sig,xf1_sig] = filter(b,a,xi,xf1_sig);
       xPrev = xp;
       xPrev_nos = xp_n;
       xPrev_sig = xp_s;

    else  %  Initalize
       [x,xf1] = filter(b,a,x);
       [x_sig,xf1_sig] = filter(b,a,xi);
       [x_nos,xf1_nos] = filter(b,a,xn);
       xPrev = zeros(size(x));
       xPrev_sig = zeros(size(xi));
       xPrev_nos = zeros(size(xn));
       slocpre = [nan, nan, nan];
    end
    %  Determine distances of COI location to microphone for weighting
    %  Find location corresponding to current time window
    [dum indx] = min(abs(spos(:,1)-(wsc-ws/2)));
    sloc = spos(indx(1),2:4);
    %  If source is active use, current location 
    if ~isnan(sloc(1))
        d = (sloc'*ones(1,samps(2))-mpos);
        slocpre = sloc;
        spflag = 0;  % clear skip processing flag
    elseif ~isnan(slocpre(1));   %  If not active, use previous location
        sloc = slocpre;
        d = (sloc'*ones(1,samps(2))-mpos);
        spflag = 0; % clear skip processing flag
    else  % If previous location not active (i.e. first frame)
        xp = zeros(length(x),samps(2)); 
        xp_n = zeros(length(xn),samps(2));
        xp_s = zeros(length(xi),samps(2));
        y = [y; zeros(length(x),1)];  % pad with zeros
        totclosemic = [totclosemic; zeros(length(x),1)];
        sigclosemic = [sigclosemic; zeros(length(xi),1)];
        nosclosemic = [nosclosemic; zeros(length(xn),1)];
        nos_beam = [nos_beam; zeros(length(xn),1)];
        sig_beam = [sig_beam; zeros(length(xi),1)];
        spflag = 1;  % Set skip processing flag

    end
    %  If valid location is present, beamform on that location
    if spflag == 0
        ar = arweights(sqrt(sum(d.^2)));
        [nscl, lmax] = max(ar);
        arw = (ar/nscl(1)).^wp;
        ww = ones(length(x),1)*arw;
        xp = x.*ww;
        xp_n = xn.*ww;
        xp_s = xi.*ww;
        
        ytemp = dsb(xp, xPrev, fs, sloc', mpos, c);
        ytemp1 = dsb(xp_n, xPrev_nos, fs, sloc', mpos, c);
        ytemp2 = dsb(xp_s, xPrev_sig, fs, sloc', mpos, c);
        y = [y; ytemp/sum(arw)];
        nos_beam =[nos_beam;ytemp1/sum(arw)];
        sig_beam =[sig_beam;ytemp2/sum(arw)];
        totclosemic = [totclosemic; x(:,lmax(1))];
        sigclosemic = [sigclosemic; xi(:,lmax(1))];
        nosclosemic = [nosclosemic; xn(:,lmax(1))];
       
    end
    %  Update for next window of data
    slocpre= sloc;  %  Update Previous Location
    wsc = wsc + ws; %  Update time window end point in seconds
    segstart = segend + 1;  % Update segement beginging
    segend = round(wsc*fs); %  Update segment end
    loopcount = loopcount + 1;
end


    sigclosemic = filter(b,a,sigclosemic);
    nosclosemic = filter(b,a,nosclosemic);
    tsig = [0:length(sigclosemic)-1]/fs;
    tbf = [0:length(y)-1]/fs;
    tcm = [0:length(totclosemic)-1]/fs;
    
figure
    plot(tsig,sigclosemic,'g')
    xlabel('Seconds')
    title('COI Signal (green)')
    hold on
    soundsc(sigclosemic,fs)
    pause(tsig(end)+1)
    plot(tcm,totclosemic,'r')
    xlabel('Seconds')
    title('COI Signal (green), Closest Mic in noise Signal (red)')
    soundsc(totclosemic,fs)
    pause(tcm(end)+1)
    plot(tbf,y,'b')
    xlabel('Seconds')
    title('COI Signal (green), Closest Mic Signal (red), Beamformed Signal (blue)')
    soundsc(y,fs)
    hold off
    
    %estimating the SNR value of the closest mic
    rmssig = mean(std(sigclosemic).^2);
    rmsnos = mean(std(nosclosemic).^2);
    snr_out = 10*log10(rmssig/(rmsnos+eps));
    disp(['1. SNR of closest mic is ' num2str(snr_out) 'dB']);
    
    %estimating the SNR value of the beamformed signals
    rmssigb = mean(std(sig_beam).^2);
    rmsnosb = mean(std(nos_beam).^2);
    snrb_out = 10*log10(rmssigb/(rmsnosb+eps));
    disp(['2. SNR of beamformed signal is ' num2str(snrb_out) 'dB']);
    
    %estimating the intelligibility for the closest mic
    [sigc, len, trimpts] = rmsilence(sigclosemic,fs);
    [sii_cm,tax] = intel(sigc,nosclosemic(trimpts),fs,100e-3);
    sii_m = mean(sii_cm);
    sii_s = std(sii_cm);
    disp(['3. Mean Intelligibility for closest mic is ' num2str(sii_m) ]);
    disp(['Standard deviation of Intelligibility for closest mic is ' num2str(sii_s) ]);
    %estimating the intelligibility for the Beamformed signals
    [siib,taxb] = intel(sig_beam(trimpts),nos_beam(trimpts),fs,100e-3);
    siib_m = mean(siib);
    siib_s = std(siib);
    disp(['4. Mean Intelligibility for beamformed signal is ' num2str(siib_m) ]);
    disp(['Standard deviation of Intelligibility for beamformed signal is ' num2str(siib_s) ]);
    % plot the speech intelligibility index of the closest mic and
    % beamformed signals.
    figure;
    plot(tax,sii_cm,'b');
    hold on
    plot(taxb,siib,'r');
    xlabel('time in seconds');
    ylabel('speech intelligibility index');
    title('Intelligibility of closest mic signals (blue) and Beamformed signals (red)');
%     soundsc(sig_beam+nos_beam,fs); % plays the seperately beamformed signals together
