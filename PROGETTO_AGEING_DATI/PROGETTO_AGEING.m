% Progetto - Chiari
clear
close all
clc

%% Questions & Tips
% 3) Lunghezza windows
% 4) Scelta features (dominio tempi e frequenze)
% 5) Analisi e classificazione su ogni soggetto e tra soggetti diversi
% 6) Mettere i 3 accelerometri insieme e valutare scelte di un
%    accelerometro piuttosto che un altro e/o errori
% 7) Mettere anche ECG e PPG insieme e classificare

%% Preprocessing
% IMU on the ankle, ECG on the chest, PPG on the wrist
%subject=["alessandra","alex","chiara","claudia","daniele","elena","federica","francesco","gabriele","gaetano","giorgia","luca","germano","massimo","matteo"]; % manca Sophie
% subject = ["ALE","ALEX","CHIARA","CLAUDIA","DANIELE_PA","ELENA","FEDERICA","FRANCESCO","MATTEO_G","GAETANO","GIORGIA","LUCA","DANIELE_G","MASSIMO","MATTEO_F"]; % MANCA SOPHIE
% sensor = ["IMU","ECG","PPG"];

%%
subject = {"ALE","ALEX","CHIARA","CLAUDIA","DANIELE_PA","ELENA","FEDERICA","FRANCESCO","MATTEO_G","GAETANO","GIORGIA","LUCA","DANIELE_G","MASSIMO","MATTEO_F"};
[indx,tf] = listdlg('ListString',subject);
% subject = str2double(subject);
subject = convertContainedStringsToChars(subject);
sensor = ["IMU","ECG","PPG"];
%%
n=length(subject);
for i=1:n
    for j=1:3
        stringa=strcat(sensor(j),"_",subject(i),'.mat');
        dati{i,j}=importdata(stringa);
    end
end
FS=128;

tempi_task = zeros(n,10);
for i=1:n
    [date_min(i), date_max(i)] = istanteiniziofine(dati{i,1}, dati{i,2}, dati{i,3}, FS, i);
    [dati{i,1}, dati{i,2}, dati{i,3}] = tempo_interesse(dati{i,1}, dati{i,2}, dati{i,3}, date_min(i), date_max(i));
    dati{i,1}.imu_Timestamp_Unix_CAL = [0:length(dati{i,1}.imu_Timestamp_Unix_CAL)-1]/FS;
    dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL = [0:length(dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL)-1]/FS;
    dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL = [0:length(dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL)-1]/FS;
    tempi_task(i,:) = divisione_task(dati{i,1}, FS, i); % casi particolari per Luca
end

for i=1:n % Solo per classificazione, non per analisi tra IMU, ECG e PPG
    [dati{i,1}, dati{i,2}, dati{i,3}] = filtraggio(dati{i,1}, dati{i,2}, dati{i,3}, FS);
end

fields={'S_83B4_ECG_ECG_EMG_Status1_CAL', 'S_83B4_ECG_ECG_EMG_Status2_CAL', 'S_83B4_ECG_ECG_LA_RA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_LA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_RA_24BIT_CAL', 'S_83B4_ECG_ECG_Vx_RL_24BIT_CAL'};
for i=1:n
    dati_acc{i,1}=dati{i,1};
    dati_acc{i,2}=rmfield(dati{i,2},fields);
    dati_acc{i,3}=rmfield(dati{i,3},"S_COD4_PPG_PPG_A13_CAL");
end
%%
%ECG_DATA_EXTRACTION
for i=1:size(dati,1)
ecg_data=dati{i,2};
LA_RA=ecg_data.S_83B4_ECG_ECG_LA_RA_24BIT_CAL;
LL_LA=ecg_data.S_83B4_ECG_ECG_LL_LA_24BIT_CAL;
LL_RA=ecg_data.S_83B4_ECG_ECG_LL_RA_24BIT_CAL;
%time_total=datetime(ecg_data.S_83B4_ECG_Timestamp_Unix_CAL,'convertfrom','posixtime','timezone','Europe/Rome');
time_total=ecg_data.S_83B4_ECG_Timestamp_Unix_CAL;
val={[LA_RA LL_LA LL_RA time_total']};
ECG_SUB{i,1}=val;
    if tempi_task(i,10)>time_total(end)*128
        tempi_task(i,10)=time_total(end)*128;
        kk(i)=i
    end
zz=intersect(round(time_total*128,-1),round(tempi_task(i,:),-1));
time_refe{i,1}=zz
end
%     figure()
%     subplot(3,1,1)
%     plot(time_s,LL_RA)
%     legend("LL RA")
%     subplot(3,1,2)
%     plot(time_s,LA_RA)
%     legend("LA RA")
%     subplot(3,1,3)
%     plot(time_s,LL_LA)
%     legend("LL LA")

pause
%%
%ECG_DATA_PROCESSING
close all
fps=FS;
fc=40;
fc_h=0.1;
[b,a] = butter(3,fc/(fps/2),'low');
[d,c] = butter(3,fc_h/(fps/2),'high');
L=length(subject);
for i=1:L %size(subject,2)
    LA_RA_n=ECG_SUB{i,1}{1,1}(:,1);
    LL_LA_n=ECG_SUB{i,1}{1,1}(:,2);
    LL_RA_n=ECG_SUB{i,1}{1,1}(:,3);
    time_s=ECG_SUB{i,1}{1,1}(:,4);
    t=time_s(1);
    time_s=(time_s-t)/1000;
    timef=time_s*fps/1000;
    LA_RA_n=LA_RA_n-mean(LA_RA_n);
    LL_LA_n=LL_LA_n-mean(LL_LA_n);
    LL_RA_n=LL_RA_n-mean(LL_RA_n);
    Len=length(time_s);
    %LL_RA=detrend(LL_RA(1:10*fps));
    for j=10*fps:10*fps:Len-2*fps
    LL_RA_n(j+1-10*fps:j)=detrend(LL_RA_n(j+1-10*fps:j));
    LL_LA_n(j+1-10*fps:j)=detrend(LL_LA_n(j+1-10*fps:j));
    LA_RA_n(j+1-10*fps:j)=detrend(LA_RA_n(j+1-10*fps:j));
    end
%     figure()
%     subplot(3,1,1)
%     plot(time_s,LL_RA)
%     subplot(312)
%     plot(time_s,LA_RA)
%     subplot(313)
%     plot(time_s,LL_LA)
    LL_LA_n=filtfilt(b,a,LL_LA_n);
    LL_RA_n=filtfilt(b,a,LL_RA_n);
    LA_RA_n=filtfilt(b,a,LA_RA_n);
    LL_LA_n=filtfilt(d,c,LL_LA_n);
    LL_RA_n=filtfilt(d,c,LL_RA_n);
    LA_RA_n=filtfilt(d,c,LA_RA_n);
    new_val=[LL_LA_n,LL_RA_n,LA_RA_n,time_s];
    ECG_SUB_new{i,1}=new_val;
    
end
%% TASK SPLIT
for i=1:size(ECG_SUB_new,1)
    task=["walking","drinking","stairs","sleeping","situp"];
    derivation=["LL_LA","LL_RA","LA_RA"];
    for j=1:3
    signal=ECG_SUB_new{i,1}(:,j);
    [walking,drinking,stairs,sleeping,situp]=task_generator(signal,tempi_task(i,:));
    ECG_SUBsplit{i,1}.derivation(j)=struct ('walking',walking,"drinking",drinking,"stairs",stairs,"sleeping",sleeping,"situp",situp);
    end
end

%% CORR.INDEX

%close all
prompt="subject? " + ...
    ""
s=input(prompt)
% prompt="select task \n"
% task_c=input(prompt,'s')
% task_c=sprintf(task_c)
VX_sleep=(ECG_SUBsplit{s, 1}.derivation(1).sleeping+ECG_SUBsplit{s, 1}.derivation(2).sleeping+ECG_SUBsplit{s, 1}.derivation(3).sleeping)/3;
VX_sleep=VX_sleep(2*fc:size(VX_sleep,1)-2*fc);
half=round(length(VX_sleep)/2,-1);
first_part_sleep=VX_sleep(fc:half);
LenSle=length(first_part_sleep);
second_part_sleep=VX_sleep(end:-1:length(VX_sleep)-LenSle+1);
[R,P]=corrcoef(first_part_sleep,second_part_sleep);

freq=[1:length(ECG_SUB_new{s,1}(:,4))]*fc/length(ECG_SUB_new{s,1}(:,4));
freq_s=[1:size(VX_sleep,1)]*fc/size(VX_sleep,1);
%VX_sleep=(ECG_SUBsplit{s, 1}.derivation(1).sleeping+ECG_SUBsplit{s, 1}.derivation(2).sleeping+ECG_SUBsplit{s, 1}.derivation(3).sleeping)/3;
LS=length(VX_sleep);
VX_sleepf=fft(VX_sleep,LS);

VX_walk=(ECG_SUBsplit{s, 1}.derivation(1).walking+ECG_SUBsplit{s, 1}.derivation(2).walking+ECG_SUBsplit{s, 1}.derivation(3).walking)/3;
VX_walk=VX_walk(30*FS:30*FS+LS);
VX_walkf=fft(VX_walk,LS);

VX_stairs=(ECG_SUBsplit{s, 1}.derivation(1).stairs+ECG_SUBsplit{s, 1}.derivation(2).stairs+ECG_SUBsplit{s, 1}.derivation(3).stairs)/3;
VX_stairs=VX_stairs(30*FS:30*FS+LS);
VX_stairsf=fft(VX_stairs,LS);

VX_situp=(ECG_SUBsplit{s, 1}.derivation(1).situp+ECG_SUBsplit{s, 1}.derivation(2).situp+ECG_SUBsplit{s, 1}.derivation(3).situp)/3;
VX_situpf=fft(VX_situp);

VX_drink=(ECG_SUBsplit{s, 1}.derivation(1).drinking+ECG_SUBsplit{s, 1}.derivation(2).drinking+ECG_SUBsplit{s, 1}.derivation(3).drinking)/3;
VX_drinkf=fft(VX_drink);

VX_taskF={VX_walkf,VX_drinkf,VX_stairsf,VX_sleepf,VX_situpf}
VX_task={VX_walk,VX_drink,VX_stairs,VX_sleep,VX_situp}

figure()
for i=1:size(task,2)
freq_t=[1:length(VX_taskF{i})]*fps/length(VX_taskF{i})
freq_task{i,1}={freq_t};
log_VX=20*log10(abs(VX_taskF{i}));
subplot(5,1,i)
stem(freq_t(1:round(end/2)),log_VX(1:round(end/2)),'filled')
legend(task(i))
ylim([0,100])
end
P
R
s
%%
% WN1=[0.1 15]/(FS/2)
% WN2=[15 30]/(FS/2)
% WN3=[30 45]/(FS/2)
% [b0,a0]=butter(5,WN1,'bandpass')
% [b1,a1]=butter(5,WN2,'bandpass')
% [b2,a2]=butter(5,WN3,'bandpass')
% for i=1:size(task,2)
%     vx_temp=VX_task{:,i};
%     freq_t=freq_task{i,1}{1,:};
%     VX_00_15=filtfilt(b0,a0,vx_temp);
%     VX_15_30=filtfilt(b1,a1,vx_temp);
%     VX_30_45=filtfilt(b2,a2,vx_temp);
%     VX_30_45f=fft(VX_30_45);
%     VX_15_30f=fft(VX_15_30);
%     VX_00_15f=fft(VX_00_15);
%     figure(i)
%     title(task(i))
%     subplot(3,1,1)
%     plot(freq_t,20*log10(abs(VX_00_15f)))
%     subplot(3,1,2)
%     plot(freq_t,20*log10(abs(VX_15_30f)))
%     subplot(3,1,3)
%     plot(freq_t,20*log10(abs(VX_30_45f)))
% end
% s
%%
close all
for i=1:size(task,2)
    DELTAF=length(freq_task{i,1}{1,:})/FS;
    FREQ_delta01_15=[0.1:1/DELTAF:15];
    k0115=length(FREQ_delta01_15);
    VX_01_15=VX_taskF{:,i}(1:k0115);
    VX_01_15s{i,1}=VX_01_15;
    VX_01_15=20*log10(abs(VX_01_15));
    
    FREQ_delta15_30=[15:1/DELTAF:30];
    k15_30=length(FREQ_delta15_30);
    VX_15_30=VX_taskF{:,i}(k0115:k0115+k15_30-1);
    VX_15_30s{i,1}=VX_15_30;
    VX_15_30=20*log10(abs(VX_15_30));
    
    
    FREQ_delta30_45=[30:1/DELTAF:45];
    k30_45=length(FREQ_delta30_45);
    VX_30_45=VX_taskF{:,i}(k0115+k15_30:k0115+k15_30+k30_45-1);
    VX_30_45s{i,1}=VX_30_45;
    VX_30_45=20*log10(abs(VX_30_45));
    
    figure(i)
    plot(FREQ_delta01_15,VX_01_15)
    hold on
    plot(FREQ_delta15_30,VX_15_30)
    hold on
    plot(FREQ_delta30_45,VX_30_45)
    title(task(i))

end
%%
[RWS15_30,PWS15_30]=corrcoef(abs(VX_15_30s{1,1}),abs(VX_15_30s{4,1}))
[RWS01_15,PWS01_15]=corrcoef(abs(VX_01_15s{1,1}),abs(VX_01_15s{4,1}))
[RWS30_45,PWS30_45]=corrcoef(abs(VX_30_45s{1,1}),abs(VX_30_45s{4,1}))
%%
[RWSt15_30,PWSt15_30]=corrcoef(abs(VX_15_30s{1,1}),abs(VX_15_30s{3,1}))
[RWSt01_15,PWSt01_15]=corrcoef(abs(VX_01_15s{1,1}),abs(VX_01_15s{3,1}))
[RWSt30_45,PWSt30_45]=corrcoef(abs(VX_30_45s{1,1}),abs(VX_30_45s{3,1}))
%%
for i=1:length(VX_30_45s{4,1})
    EW=sum(abs(VX_30_45s{4,1}(i).^2))
end
powbp30_45w = bandpower(VX_30_45s{1,1},FS,[30 45])
powbp15_30w = bandpower(VX_30_45s{1,1},FS,[15 30])
powbp01_15w = bandpower(VX_30_45s{1,1},FS,[1 15])

powbp30_45st = bandpower(VX_30_45s{3,1},FS,[30 45])
powbp15_30st = bandpower(VX_30_45s{3,1},FS,[15 30])
powbp01_15st = bandpower(VX_30_45s{3,1},FS,[1 15])
%%
for i=1:size(task,2)
    tt_task = [1 : length(VX_task{1,i})];
    
    % PULIZIA DEL SEGNALE
    smoothECG = sgolayfilt(VX_task{1,i},7,21);
    figure()
    subplot(211)
    plot(tt_task,VX_task{1,i},'b',tt_task,smoothECG,'r')
    grid on
    axis tight
    xlabel('Samples')
    ylabel('Voltage(mV)')
    legend('Noisy ECG Signal','Filtered Signal')
    title('Filtering Noisy ECG Signal of ' + task(i));
    
    subplot(212)
    plot(tt_task,smoothECG)
    grid on
    axis tight
    title('ECG ' + task(i))

    [R_pks,R_locs] = findpeaks(smoothECG,'MinPeakDistance',0.5 * FS); 
    figure
    plot(tt_task,smoothECG)
    hold on
    plot(tt_task(R_locs),smoothECG(R_locs),'*')
    
    RR_ist = (R_locs(2:end) - R_locs(1:end-1)) * 1/FS;
    t_RR_ist = cumsum(RR_ist); 
    t_RR_ist = t_RR_ist - t_RR_ist(1) * ones(size(t_RR_ist(1)));
    HR_ECG = mean(60 ./ RR_ist)
    SDNN_ECG = std(RR_ist) *1000;
    RMSSD_ECG = sqrt(mean((RR_ist(2:end) - RR_ist(1:end-1)).^2)) * 1000;
    RR_ms = RR_ist * 1000; 
    RR_demean = RR_ms - mean(RR_ms)*ones(size(RR_ms))
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
    title(task(i))
    
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
    title(task(i))
    LFHF_ratio_ECG = aLF/aHF
end
%% poincarè
% x = RR_ms; 
% x(end)=[];  
% y = RR_ms; 
% y(1) = [];
% L = length(x);
% SD1_RR = sqrt((1/L) * sum(((x- y)- mean(x -y)).^2)/2);
% SD2_RR = sqrt((1/L) * sum(((x + y) - mean(x + y)).^2)/2);
% 
% Ellipse center
% xc = mean(RR_ms);
% yc = mean(RR_ms);
% 
% Ellipse axes
% maj_ax = (2*SD2_RR);
% min_ax = (2*SD1_RR);
% 
% Ellipse draw
% alfa = deg2rad(45);
% [X_ell,Y_ell] = ellipsedraw(maj_ax,min_ax,xc,yc,alfa);
% 
% figure
% plot(RR_ms(1:end-1),RR_ms(2:end),'o')
% hold on
% plot(RR_ms,RR_ms,'-')
% plot(X_ell,Y_ell,'linewidth',2)
% xlabel('RR_i ms')
% ylabel('RR_{(i+1)} ms')
%%
%     figure()
%     subplot(3,1,1)
%     plot(time_s,LL_RA_n)
%     legend("LL RA")
%     subplot(3,1,2)
%     plot(time_s,LA_RA_n)
%     legend("LA RA")
%     subplot(3,1,3)
%     plot(time_s,LL_LA_n)
%     legend("LL LA")
    %%
%     walk_ecg=[LL_LA(3*fps:90*fps),LL_RA(3*fps:90*fps),LA_RA(3*fps:90*fps)];
%     figure(3)
%     for i=1:3
%     subplot(3,1,i)
%     plot(time_s(3*fps:90*fps),walk_ecg(:,i))
%     end
%     %%
%     walk=[3*fps:90*fps];
%     LL_RAf=fft(LL_RA(walk));
%     LL_LAf=fft(LL_LA(walk));
%     LA_RAf=fft(LA_RA(walk));
%     walk_ecgf=[LL_LAf,LL_RAf,LA_RAf];
%     lab_w=["LL LA","LL RA","LA RA"]
%     S=size(walk_ecgf,1);
%     freqw=(1:S)*fps/S;
%     figure(4)
%     for i=1:3
%     subplot(3,1,i)
%     plot(freqw,walk_ecgf(:,i))
%     legend(lab_w(i))
%     end
% 
% 
% %%
% [qrspeaks_ll_la,locs_ll_la] = findpeaks(LL_LA,time_s,'MinPeakHeight',0.2, 'MinPeakDistance',0.150);
% [qrspeaks_ll_ra,locs_ll_ra] = findpeaks(abs(LL_RA),time_s,'MinPeakHeight',0.2,'MinPeakDistance',0.150);
% [qrspeaks_la_ra,locs_la_ra] = findpeaks(LA_RA,time_s,'MinPeakHeight',0.2,'MinPeakDistance',0.150);
% %%
% X_acc=data.S_83B4_ECG_Accel_WR_X_CAL;
% Y_acc=data.S_83B4_ECG_Accel_WR_Y_CAL;
% Z_acc=data.S_83B4_ECG_Accel_WR_Z_CAL;
% [peak_x,loc_x]=findpeaks(X_acc,"MinPeakHeight",5,"MinPeakDistance",30*fps)
% locX_sp=loc_x/(fps)
% if locX_sp(1)>3
%         locX_sw=3;
%         locX_s=[locX_sw,locX_sp'];
% end
% if locX_s(2)>120
%     Tw=locX_sp(length(locX_sp))
%     for i=3:length(locX_s)
%         locX_s(i)==locX_s(i-1);
%     end
%    locX_s(2)==120;
%    locX_s=[locX_s, Tw];
% end
% cc=length(locX_s)
% %%
% %%
% figure(5)
% subplot(3,1,1)
% plot(time_s,X_acc)
% title("X axis")
% subplot(3,1,2)
% plot(time_s,Y_acc)
% title("Y axis")
% subplot(3,1,3)
% plot(time_s,Z_acc)
% title("Z axis")

% for i=1:size(dati,1)
% ECG_SUB{}.
% ecg_cell(i,4)=time_sub
% task_ecg=time_sub(time_task(i,:))
% end
%%
% ppg_signal=dati{4,3};
% ppg_value=ppg_signal.S_COD4_PPG_PPG_A13_CAL;
% T=size(ppg_value,1);
% ppg_signal=ppg_value - mean(ppg_value);
% [b,a]=butter(3,5/(128/2));
% ppg_value=filtfilt(b,a,ppg_value);
% ppg_value=detrend(ppg_value)
% t=[1:T]/FS;
% figure()
% plot(t(800*FS:840*FS),ppg_value(800*FS:840*FS));
% hold on
% figure()
% plot(t,ppg_value);
%%
% ppg_signal_f=load("PPG_Giorgia.mat")
% ppg_value=ppg_signal_f.S_COD4_PPG_PPG_A13_CAL;
% T=size(ppg_value,1);
% ppg_signal=ppg_value - mean(ppg_value);
% [b,a]=butter(3,5/(128/2));
% ppg_value=filtfilt(b,a,ppg_value);
% ppg_value=detrend(ppg_value)
% t=[1:T]/FS;
% figure()
% plot(t(720*FS:740*FS),ppg_value(720*FS:740*FS));
% title("gio ppg")
% hold on
% figure()
% plot(t,ppg_value);
% title("gio ppg")
% plot(t(600*FS:700*FS),ppg_value(600*FS:700*FS));
% %
% ppg_signal_f=load("federica_PPG.mat")
% ppg_value=ppg_signal_f.S_COD4_PPG_PPG_A13_CAL;
% T=size(ppg_value,1);
% ppg_signal=ppg_value - mean(ppg_value);
% [b,a]=butter(3,5/(128/2));
% ppg_value=filtfilt(b,a,ppg_value);
% ppg_value=detrend(ppg_value)
% t=[1:T]/FS;
% figure()
% plot(t(700*FS:720*FS),ppg_value(700*FS:720*FS));
% title("fede ppg")
% hold on
% figure()
% plot(t,ppg_value);
% title("fede ppg")
% plot(t(600*FS:700*FS),ppg_value(600*FS:700*FS));