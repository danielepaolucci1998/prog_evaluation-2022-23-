clear all 
close all
clc

sub1_dir = '.\Rec'; 
load(fullfile(sub1_dir,'ECG.mat')); 
load(fullfile(sub1_dir,'PPG.mat')); 

% Initialize signals 
% ECG
N_ECG = length(ECG); 
fs_ECG = 256; % [Hz]
Ts_ECG = 1/fs_ECG; % [s]
t_ECG = (0:1:N_ECG-1) * Ts_ECG; 

% PPG
N_PPG = length(PPG); 
fs_PPG = 64; % [Hz]
Ts_PPG = 1/fs_PPG; % [s]
t_PPG = (0:1:N_PPG-1) * Ts_PPG; 

% Visualize signals in the same figure
figure
a1 = subplot(2,1,1); 
plot(t_ECG,ECG)
xlabel('Time [s]'), ylabel('ECG [mV]')
a2 = subplot(2,1,2); 
plot(t_PPG,PPG)
xlabel('Time [s]'), ylabel('PPG [a.u.]')
linkaxes([a1 a2],'x')

% ECG preprocessing 
ordr     = 7;
ft1      = 30; %[Hz], frequency cutoff
Wn       = ft1/(fs_ECG / 2); %Normalized cutoff frequency
[b,a]    = butter(ordr,Wn);
ECG_filt = filtfilt(b,a,ECG);


%% Find R peaks from ECG signal 
% Minimum distance = 0.5 s
[R_pks,R_locs] = findpeaks(ECG_filt,'MinPeakDistance',0.5 * fs_ECG); 
figure
plot(t_ECG,ECG_filt)
hold on
plot(t_ECG(R_locs),ECG_filt(R_locs),'*')

RR_ist = (R_locs(2:end) - R_locs(1:end-1)) * Ts_ECG;
t_RR_ist = cumsum(RR_ist); 
t_RR_ist = t_RR_ist - t_RR_ist(1) * ones(size(t_RR_ist(1)));

%% Find systolic foot 
[sysfeet_pks,sysfeet_locs] = findpeaks(-PPG,'MinPeakDistance',0.5 * fs_PPG); 
figure
plot(t_PPG,PPG)
hold on
plot(t_PPG(sysfeet_locs),PPG(sysfeet_locs),'*')

IBI_ist = (sysfeet_locs(2:end) - sysfeet_locs(1:end-1)) * Ts_PPG;
t_IBI_ist = cumsum(IBI_ist);
t_IBI_ist = t_IBI_ist - t_IBI_ist(1) * ones(size(t_IBI_ist(1)));

figure
plot(t_RR_ist,RR_ist)
hold on
plot(t_IBI_ist,IBI_ist)

% Calculate HRV parameters
% HR [bpm]
HR_ECG = mean(60 ./ RR_ist); % [bpm]
HR_PPG = mean(60 ./ IBI_ist); % [bpm]

% SDNN [ms]
SDNN_ECG = std(RR_ist) *1000; 
SDNN_PPG = std(IBI_ist) * 1000;

% RMSSD [ms]
RMSSD_ECG = sqrt(mean((RR_ist(2:end) - RR_ist(1:end-1)).^2)) * 1000;
RMSSD_PPG = sqrt(mean((IBI_ist(2:end) - IBI_ist(1:end-1)).^2)) * 1000;

% Frequency domain 
% LF/HF
RR_ms = RR_ist * 1000; 
RR_demean = RR_ms - mean(RR_ms)*ones(size(RR_ms));
% Detrend
RR_detrend = detrend(RR_demean,'linear');
% Interpolation
F_resample = 4; 
T_resample = 1/F_resample; 
t_res = t_RR_ist(1):T_resample:t_RR_ist(end); 
RR_ist_resample = interp1(t_RR_ist,RR_detrend,t_res);
% High pass filter
ordr     = 6;
ft       = 0.02; %[Hz] Cutoff frequency
Wn_HP    = ft/(F_resample/2); %Normalized cutoff frequency
[b, a]   = butter(ordr,Wn_HP,'high');
RR_HP   = filtfilt(b,a,RR_ist_resample);
% Low pass filter
ordr     = 6;
ft       = 0.4; %[Hz] Cutoff frequency
Wn_LP       = ft/(F_resample/2); %Normalized cutoff frequency
[b, a]   = butter(ordr,Wn_LP);
RR_filt = filtfilt(b,a,RR_HP);

% window = 5*60*F_resample;
L = length(RR_filt); 
NFFT = 2^nextpow2(L); % Vollmer M. 2015 "A robust, simple and reliable measure of HRV using relative RR intervals"
[PSD,F] = pwelch(RR_filt,[],[],NFFT,4);

figure
plot(F,PSD)

LF = [0.04 0.15];
HF = [0.15 0.4];

iLF = (round(F,2) >= LF(1)) & (round(F,2) <= LF(2));
aLF  = trapz(F(iLF),PSD(iLF))

iHF = (round(F,2) >= HF(1)) & (round(F,2) <= HF(2));
aHF  = trapz(F(iHF),PSD(iHF))

i_TP = (round(F,2) >= LF(1)) & (round(F,2) <= HF(2)); 
TP   = trapz(F(i_TP),PSD(i_TP))

% Normalized values of the spectral bands
nu_LF = aLF / TP * 100; 
nu_HF = aHF / TP * 100;

figure
plot(F,PSD)
xlabel('Frequency [Hz]'), ylabel('PSD [ms^2]')
hold on 
area(F(iLF),PSD(iLF))
area(F(iHF),PSD(iHF))

LFHF_ratio_ECG = aLF/aHF

IBI_ms = IBI_ist * 1000;
IBI_demean = IBI_ms - mean(IBI_ms)*ones(size(IBI_ms));
% Detrend
IBI_detrend = detrend(IBI_demean,'linear');
% Interpolation
F_resample = 4; 
T_resample = 1/F_resample; 
t_res = t_IBI_ist(1):T_resample:t_IBI_ist(end); 
IBI_ist_resample = interp1(t_IBI_ist,IBI_detrend,t_res);
% High pass filter
ordr     = 6;
ft       = 0.02; %[Hz] Cutoff frequency
Wn_HP    = ft/(F_resample/2); %Normalized cutoff frequency
[b, a]   = butter(ordr,Wn_HP,'high');
IBI_HP   = filtfilt(b,a,IBI_ist_resample);
% Low pass filter
ordr     = 6;
ft       = 0.4; %[Hz] Cutoff frequency
Wn_LP       = ft/(F_resample/2); %Normalized cutoff frequency
[b, a]   = butter(ordr,Wn_LP);
IBI_filt = filtfilt(b,a,IBI_HP);

window = 5*60*F_resample;
L = length(IBI_filt); 
NFFT = 2^nextpow2(L); % Vollmer M. 2015 "A robust, simple and reliable measure of HRV using relative RR intervals"
[PSD,F] = pwelch(IBI_filt,[],[],NFFT,4);

figure
plot(F,PSD)

iLF = (round(F,2) >= LF(1)) & (round(F,2) <= LF(2));
aLF  = trapz(F(iLF),PSD(iLF))

iHF = (round(F,2) >= HF(1)) & (round(F,2) <= HF(2));
aHF  = trapz(F(iHF),PSD(iHF))

i_TP = (round(F,2) >= LF(1)) & (round(F,2) <= HF(2)); 
TP   = trapz(F(i_TP),PSD(i_TP))

% Normalized values of the spectral bands
nu_LF = aLF / TP * 100; 
nu_HF = aHF / TP * 100;

figure
plot(F,PSD)
xlabel('Frequency [Hz]'), ylabel('PSD [ms^2]')
hold on 
area(F(iLF),PSD(iLF))
area(F(iHF),PSD(iHF))

LFHF_ratio_PPG = aLF/aHF

% Poincaré Plot

% x-axis: RR_ist(i), y-axis: RR_ist(i+1)
x = RR_ms; 
x(end)=[];  
y = RR_ms; 
y(1) = [];
L = length(x);
SD1_RR = sqrt((1/L) * sum(((x- y)- mean(x -y)).^2)/2);
SD2_RR = sqrt((1/L) * sum(((x + y) - mean(x + y)).^2)/2);

%Ellipse center
xc = mean(RR_ms);
yc = mean(RR_ms);

%Ellipse axes
maj_ax = (2*SD2_RR);
min_ax = (2*SD1_RR);

%Ellipse draw
alfa = deg2rad(45);
[X_ell,Y_ell] = ellipsedraw(maj_ax,min_ax,xc,yc,alfa);

figure
plot(RR_ms(1:end-1),RR_ms(2:end),'o')
hold on
plot(RR_ms,RR_ms,'-')
plot(X_ell,Y_ell,'linewidth',2)
xlabel('RR_i ms')
ylabel('RR_{(i+1)} ms')

x = IBI_ms; 
x(end)=[];  
y = IBI_ms; 
y(1) = [];
L = length(x);
SD1_IBI = sqrt((1/L) * sum(((x - y)- mean(x -y)).^2)/2);
SD2_IBI = sqrt((1/L) * sum(((x + y) - mean(x + y)).^2)/2);

%Ellipse center
xc = mean(IBI_ms);
yc = mean(IBI_ms);

%Ellipse axes
maj_ax = (2*SD2_IBI);
min_ax = (2*SD1_IBI);

%Ellipse draw
alfa = deg2rad(45);
[X_ell,Y_ell] = ellipsedraw(maj_ax,min_ax,xc,yc,alfa);

figure
plot(IBI_ms(1:end-1),IBI_ms(2:end),'o')
hold on
plot(IBI_ms,IBI_ms,'-')
plot(X_ell,Y_ell,'linewidth',2)
xlabel('IBI_i ms')
ylabel('IBI_{(i+1)} ms')