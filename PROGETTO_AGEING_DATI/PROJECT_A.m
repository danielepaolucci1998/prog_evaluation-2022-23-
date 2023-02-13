 %% PROGETTO

clear all
close all
clc

%% Questions & Tips

% 3) Lunghezza windows
% 4) Scelta features (dominio tempi e frequenze)
% 5) Analisi e classificazione su ogni soggetto e tra soggetti diversi
% 6) Mettere i 3 accelerometri insieme e valutare scelte di un
%    accelerometro piuttosto che un altro e/o errori
% 7) Mettere anche ECG e PPG insieme e classificare


%%
%% Preprocessing
%%
% IMU on the ankle, ECG on the chest, PPG on the wrist

subject = ["ALE","ALEX","CHIARA","CLAUDIA","DANIELE_PA","DANI_G","ELENA","FEDERICA","FRANCESCO","GAETANO","GIORGIA","LUCA","MASSIMO","MATTEO_F","MATTEO_G","SOPHIE"];

% Pat = 'Select a Subject : ';
% subject = input(Pat);

sensor = ["IMU","ECG","PPG"];
n = length(subject);

for i = 1:n
    for j = 1:3
        stringa = strcat(sensor(j),"_",subject(i),'.mat');
        dati{i,j} = importdata(stringa);
    end
end
FS = 128;

tempi_task = zeros(n,10);

for i = 1:n
    [date_min(i), date_max(i)] = istanteiniziofine(dati{i,1}, dati{i,2}, dati{i,3}, FS, i);
    [dati{i,1}, dati{i,2}, dati{i,3}] = tempo_interesse(dati{i,1}, dati{i,2}, dati{i,3}, date_min(i), date_max(i));
    dati{i,1}.imu_Timestamp_Unix_CAL = [0:length(dati{i,1}.imu_Timestamp_Unix_CAL)-1]/FS;
    dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL = [0:length(dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL)-1]/FS;
    dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL = [0:length(dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL)-1]/FS;
    tempi_task(i,:) = divisione_task(dati{i,1}, FS, i); % casi particolari per Luca
end

for i = 1:n % Solo per classificazione, non per analisi tra IMU, ECG e PPG
    [dati{i,1}, dati{i,2}, dati{i,3}] = filtraggio(dati{i,1}, dati{i,2}, dati{i,3}, FS);
end

fields = {'S_83B4_ECG_ECG_EMG_Status1_CAL', 'S_83B4_ECG_ECG_EMG_Status2_CAL', 'S_83B4_ECG_ECG_LA_RA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_LA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_RA_24BIT_CAL', 'S_83B4_ECG_ECG_Vx_RL_24BIT_CAL'};

for i = 1:n
    dati_acc{i,1} = dati{i,1};
    dati_acc{i,2} = rmfield(dati{i,2},fields);
    dati_acc{i,3} = rmfield(dati{i,3},"S_COD4_PPG_PPG_A13_CAL");
end



%%
%% ECG
%%

% ECG_DATA_EXTRACTION

for i = 1:size(dati,1)
    ecg_data = dati{i,2};
    LA_RA = ecg_data.S_83B4_ECG_ECG_LA_RA_24BIT_CAL;
    LL_LA = ecg_data.S_83B4_ECG_ECG_LL_LA_24BIT_CAL;
    LL_RA = ecg_data.S_83B4_ECG_ECG_LL_RA_24BIT_CAL;
    % time_total = datetime(ecg_data.S_83B4_ECG_Timestamp_Unix_CAL,'convertfrom','posixtime','timezone','Europe/Rome');
    time_total = ecg_data.S_83B4_ECG_Timestamp_Unix_CAL;
    val = {[LA_RA LL_LA LL_RA time_total']};
    ECG_SUB{i,1} = val;
    
        if tempi_task(i,10) > time_total(end)*128
            tempi_task(i,10) = time_total(end)*128;
            kk(i) = i;
        end
    
    zz = intersect(round(time_total*128,-1),round(tempi_task(i,:),-1));
    time_refe{i,1} = zz;

end

%%
%% ECG_DATA_PROCESSING

% filtraggio a 40Hz

fps = FS;
fc = 40;
[b,a] = butter(3,fc/(fps/2),'low');

for i = 1:2 % size(subject,2)

    LA_RA_n = ECG_SUB{i,1}{1,1}(:,1);
    LL_LA_n = ECG_SUB{i,1}{1,1}(:,2);
    LL_RA_n = ECG_SUB{i,1}{1,1}(:,3);
    time_s = ECG_SUB{i,1}{1,1}(:,4);
    t = time_s(1);
    time_s = (time_s-t)/1000;
    timef = time_s*fps/1000;
    LA_RA_n = LA_RA_n-mean(LA_RA_n);
    LL_LA_n = LL_LA_n-mean(LL_LA_n);
    LL_RA_n = LL_RA_n-mean(LL_RA_n);
    L = length(time_s);
    % LL_RA = detrend(LL_RA(1:10*fps));

    for j = 10*fps:10*fps:L-2*fps
        LL_RA_n(j+1-10*fps:j) = detrend(LL_RA_n(j+1-10*fps:j)); % raddrizza ECG con intervalli di 10 sec
        LL_LA_n(j+1-10*fps:j) = detrend(LL_LA_n(j+1-10*fps:j));
        LA_RA_n(j+1-10*fps:j) = detrend(LA_RA_n(j+1-10*fps:j));
    end

%     figure()
%     subplot(3,1,1)
%     plot(time_s,LL_RA)
%     subplot(312)
%     plot(time_s,LA_RA)
%     subplot(313)
%     plot(time_s,LL_LA)

    LL_LA_n = filtfilt(b,a,LL_LA_n);
    LL_RA_n = filtfilt(b,a,LL_RA_n);
    LA_RA_n = filtfilt(b,a,LA_RA_n);
    new_val = [LL_LA_n,LL_RA_n,LA_RA_n,time_s];
    ECG_SUB_new{i,1} = new_val;

    figure()
    subplot(3,1,1)
    plot(time_s,LL_RA(i))
    legend("LL RA");
    subplot(3,1,2)
    plot(time_s,LA_RA(i))
    legend("LA RA");
    subplot(3,1,3)
    plot(time_s,LL_LA(i))
    legend("LL LA");

end


%% Fase Camminata

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

%% PPG
%%
% ppg_signal=dati{6,3};
% ppg_value=ppg_signal.S_COD4_PPG_PPG_A13_CAL;
% T=size(ppg_value,1);
% ppg_signal=ppg_value - mean(ppg_value);
% [b,a]=butter(3,5/(128/2));
% ppg_value=filtfilt(b,a,ppg_value);
% ppg_value=detrend(ppg_value)
% t=[1:T]/FS;
% figure()
% plot(t(720*FS:740*FS),ppg_value(720*FS:740*FS));
% % hold on
% figure()
% plot(t,ppg_value);
% %%
% ppg_signal_f=load("PPG_Giorgia.mat")
% ppg_value=ppg_signal_f.S_COD4_PPG_PPG_A13_CAL;
% T=size(ppg_value,1);
% ppg_signal=ppg_value - mean(ppg_value);
% [b,a]=butter(3,5/(128/2));
% ppg_value=filtfilt(b,a,ppg_value);
% %ppg_value=detrend(ppg_value)
% t=[1:T]/FS;
% figure()
% plot(t(720*FS:740*FS),ppg_value(720*FS:740*FS));
% title("gio ppg")
% % hold on
% figure()
% plot(t,ppg_value);
% title("gio ppg")
% % plot(t(600*FS:700*FS),ppg_value(600*FS:700*FS));
% %%
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
% % hold on
% figure()
% plot(t,ppg_value);
% title("fede ppg")
% % plot(t(600*FS:700*FS),ppg_value(600*FS:700*FS));















%%
%% Processing 

% t=dati_acc{6,1}.imu_Timestamp_Unix_CAL;
% accx_imu=dati_acc{6,1}.imu_Accel_WR_X_CAL;
% accy_imu=dati_acc{6,1}.imu_Accel_WR_Y_CAL;
% accz_imu=dati_acc{6,1}.imu_Accel_WR_Z_CAL;
% smv_imu=dati_acc{6,1}.imu_Accel_Tot;
% gyrx_imu=dati_acc{6,1}.imu_Gyro_X_CAL;
% gyry_imu=dati_acc{6,1}.imu_Gyro_Y_CAL;
% gyrz_imu=dati_acc{6,1}.imu_Gyro_Z_CAL;
% accx_ecg=dati_acc{6,2}.S_83B4_ECG_Accel_WR_X_CAL;
% accy_ecg=dati_acc{6,2}.S_83B4_ECG_Accel_WR_Y_CAL;
% accz_ecg=dati_acc{6,2}.S_83B4_ECG_Accel_WR_Z_CAL;
% smv_ecg=dati_acc{6,2}.S_83B4_ECG_Accel_Tot;
% gyrx_ecg=dati_acc{6,2}.S_83B4_ECG_Gyro_X_CAL;
% gyry_ecg=dati_acc{6,2}.S_83B4_ECG_Gyro_Y_CAL;
% gyrz_ecg=dati_acc{6,2}.S_83B4_ECG_Gyro_Z_CAL;
% accx_ppg=dati_acc{6,3}.S_COD4_PPG_Accel_WR_X_CAL;
% accy_ppg=dati_acc{6,3}.S_COD4_PPG_Accel_WR_Y_CAL;
% accz_ppg=dati_acc{6,3}.S_COD4_PPG_Accel_WR_Z_CAL;
% smv_ppg=dati_acc{6,3}.S_COD4_PPG_Accel_Tot;
% gyrx_ppg=dati_acc{6,3}.S_COD4_PPG_Gyro_X_CAL;
% gyry_ppg=dati_acc{6,3}.S_COD4_PPG_Gyro_Y_CAL;
% gyrz_ppg=dati_acc{6,3}.S_COD4_PPG_Gyro_Z_CAL;
%     figure()
%     subplot(411),plot(t,accx_ppg), xlabel('Time [s]'), ylabel('x-acc [m/s^2]');
%     title('Accelerations - PPG'); 
%     subplot(412),plot(t,accy_ppg), xlabel('Time [s]'), ylabel('y-acc [m/s^2]');
%     subplot(413),plot(t,accz_ppg), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'); 
%     subplot(414),plot(t,smv_ppg), xlabel('Time [s]'), ylabel('tot acc [m/s^2]');
%     figure()
%     subplot(311),plot(t,gyrx_ppg), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title('Angular velocities - PPG');
%     subplot(312),plot(t,gyry_ppg), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t,gyrz_ppg), xlabel('Time [s]'),
%     ylabel('z-ang vel [rad/s]');
%     figure()
%     subplot(411),plot(t,accx_imu), xlabel('Time [s]'), ylabel('x-acc [m/s^2]');
%     title('Accelerations - IMU'); 
%     subplot(412),plot(t,accy_imu), xlabel('Time [s]'), ylabel('y-acc [m/s^2]');
%     subplot(413),plot(t,accz_imu), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'); 
%     subplot(414),plot(t,smv_imu), xlabel('Time [s]'), ylabel('tot acc [m/s^2]');
%     figure()
%     subplot(311),plot(t,gyrx_imu), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title('Angular velocities - IMU');
%     subplot(312),plot(t,gyry_imu), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t,gyrz_imu), xlabel('Time [s]'),
%     ylabel('z-ang vel [rad/s]');
%         figure()
%     subplot(411),plot(t,accx_ecg), xlabel('Time [s]'), ylabel('x-acc [m/s^2]');
%     title('Accelerations - ECG'); 
%     subplot(412),plot(t,accy_ecg), xlabel('Time [s]'), ylabel('y-acc [m/s^2]');
%     subplot(413),plot(t,accz_ecg), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'); 
%     subplot(414),plot(t,smv_ecg), xlabel('Time [s]'), ylabel('tot acc [m/s^2]');


label = [1,2,3,4,5];
% task 4
tab4_imu=[]; tab4_ecg=[]; tab4_ppg=[];
for i=1:1
    t=dati_acc{i,1}.imu_Timestamp_Unix_CAL;
    for j=1:3
        cont = tempi_task(i,7) + 3*FS;
        k=1;
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        while cont + 8*FS < tempi_task(i,8) - 3*FS
            finestre_task4(k,:) = [cont, cont + 8*FS - 1];
            tt = t(finestre_task4(k,1):finestre_task4(k,2));
            smv = smv_sensor(finestre_task4(k,1):finestre_task4(k,2));
            feat4 = task4(tt, smv, FS, i, j, label(4));
            if j==1
                tab4_imu = [tab4_imu; feat4];
            elseif j==2 
                tab4_ecg = [tab4_ecg; feat4];
            else
                tab4_ppg = [tab4_ppg; feat4];
            end
            cont = cont+8*FS; k = k+1;
        end
    end
end