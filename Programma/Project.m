% Progetto - Chiari
clear
close all
clc

%% Questions & Tips
% 3) Lunghezza windows
% 4) Scelta features (dominio tempi e frequenze)
% standardizzazione features tra soggetti diversi (una per sensore) 
% media features, sottraggo ad ognuno e divido per dev.standard
% 5) Analisi e classificazione su ogni soggetto e tra soggetti diversi
% con classificatori diversi (su Python)


%% Preprocessing
% IMU on the ankle, ECG on the chest, PPG on the wrist
subject=["alessandra","alex","chiara","claudia","daniele","elena","federica","francesco","gabriele","gaetano","giorgia","luca","germano","massimo","matteo"]; % manca Sophie
sensor=["IMU","ECG","PPG"];
n=length(subject);
for i=1:n
    for j=1:3
        stringa=strcat(subject(i),'_',sensor(j),'.mat');
        dati{i,j}=importdata(stringa);
    end
end
FS=128;

t7 = datetime(dati{7,1}.imu_Timestamp_Unix_CAL/1000,'convertfrom','posixtime','timezone','Europe/Rome');
t11 = datetime(dati{11,1}.imu_Timestamp_Unix_CAL/1000,'convertfrom','posixtime','timezone','Europe/Rome');

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

for i=11
t_imu=dati_acc{i,1}.imu_Timestamp_Unix_CAL(tempi_task(i,1):tempi_task(i,10));
t_ecg=dati_acc{i,2}.S_83B4_ECG_Timestamp_Unix_CAL(tempi_task(i,1):tempi_task(i,10));
t_ppg=dati_acc{i,3}.S_COD4_PPG_Timestamp_Unix_CAL(tempi_task(i,1):tempi_task(i,10));
% dati_imu=dati{i,1};
% dati_ecg=dati{i,2};
% dati_ppg=dati{i,3};
% accx_imu=dati{i,1}.imu_Accel_WR_X_CAL(tempi_task(i,1):tempi_task(i,2));
% accy_imu=dati{i,1}.imu_Accel_WR_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% accz_imu=dati{i,1}.imu_Accel_WR_Z_CAL(tempi_task(i,1):tempi_task(i,2));
smv_imu=dati_acc{i,1}.imu_Accel_Tot(tempi_task(i,1):tempi_task(i,10));
% gyrx_imu=dati{i,1}.imu_Gyro_X_CAL(tempi_task(i,1):tempi_task(i,2));
% gyry_imu=dati{i,1}.imu_Gyro_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% gyrz_imu=dati{i,1}.imu_Gyro_Z_CAL(tempi_task(i,1):tempi_task(i,2));
% accx_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_X_CAL(tempi_task(i,1):tempi_task(i,2));
% accy_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% accz_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_Z_CAL(tempi_task(i,1):tempi_task(i,2));
smv_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_Tot(tempi_task(i,1):tempi_task(i,10));
% gyrx_ecg=dati_acc{i,2}.S_83B4_ECG_Gyro_X_CAL(tempi_task(i,1):tempi_task(i,2));
% gyry_ecg=dati_acc{i,2}.S_83B4_ECG_Gyro_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% gyrz_ecg=dati_acc{i,2}.S_83B4_ECG_Gyro_Z_CAL(tempi_task(i,1):tempi_task(i,2));
% accx_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_X_CAL(tempi_task(i,1):tempi_task(i,2));
% accy_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% accz_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_Z_CAL(tempi_task(i,1):tempi_task(i,2));
smv_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_Tot(tempi_task(i,1):tempi_task(i,10));
% gyrx_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_X_CAL(tempi_task(i,1):tempi_task(i,2));
% gyry_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_Y_CAL(tempi_task(i,1):tempi_task(i,2));
% gyrz_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_Z_CAL(tempi_task(i,1):tempi_task(i,2));
% [~, ind] = findpeaks(abs(gyry_ecg),'MinPeakHeight',30,'MinPeakDistance',10*FS);
    figure()
%     subplot(411),plot(t_imu,accx_imu), xlabel('Time [s]'), ylabel('x-acc [m/s^2]'), ylim([-30,30]);
%     title(['Accelerations - IMU - subj: ',num2str(i)]); 
%     subplot(412),plot(t_imu,accy_imu), xlabel('Time [s]'), ylabel('y-acc [m/s^2]'), ylim([-30,30]);
%     subplot(413),plot(t_imu,accz_imu), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'), ylim([-30,30]);
    subplot(414),plot(t_imu,smv_imu), xlabel('Time [s]'), ylabel('tot acc [m/s^2]'), ylim([-30,30]);
%     figure()
%     subplot(311),plot(t_imu,gyrx_imu), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title(['Angular velocities - IMU - subj: ',num2str(i)]');
%     subplot(312),plot(t_imu,gyry_imu), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t_imu,gyrz_imu), xlabel('Time [s]'),
%     ylabel('z-ang vel [rad/s]');
    figure()
%     subplot(411),plot(t_ecg,accx_ecg), xlabel('Time [s]'), ylabel('x-acc [m/s^2]'), ylim([-10,10]);
%     title(['Accelerations - ECG - subj: ',num2str(i)]'); 
%     subplot(412),plot(t_ecg,accy_ecg), xlabel('Time [s]'), ylabel('y-acc [m/s^2]'), ylim([-10,10]);
%     subplot(413),plot(t_ecg,accz_ecg), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'), ylim([-10,10]); 
    subplot(414),plot(t_ecg,smv_ecg), xlabel('Time [s]'), ylabel('tot acc [m/s^2]'), ylim([-10,10]);
%     figure()
%     subplot(311),plot(t_ecg,gyrx_ecg), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title(['Angular velocities - ECG - subj: ',num2str(i)]');
%     subplot(312),plot(t_ecg,abs(gyry_ecg),t_ecg(ind),abs(gyry_ecg(ind)),'*k'), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t_ecg,gyrz_ecg), xlabel('Time [s]'),
%     ylabel('z-ang vel [rad/s]');
%     figure()
%     subplot(311),plot(t_imu,gyrx_imu), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title('Angular velocities - ECG');
%     subplot(312),plot(t_imu,gyry_imu), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t_imu,gyrz_imu), xlabel('Time [s]'),
% 
% t_imu=dati_acc{i,1}.imu_Timestamp_Unix_CAL;
% t_ecg=dati_acc{i,2}.S_83B4_ECG_Timestamp_Unix_CAL;
% t_ppg=dati_acc{i,3}.S_COD4_PPG_Timestamp_Unix_CAL;
% accx_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_X_CAL;
% accy_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_Y_CAL;
% accz_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_Z_CAL;
% smv_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_Tot;
% gyrx_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_X_CAL;
% gyry_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_Y_CAL;
% gyrz_ppg=dati_acc{i,3}.S_COD4_PPG_Gyro_Z_CAL;
%     figure()
%     subplot(411),plot(t_ppg,accx_ppg), xlabel('Time [s]'), ylabel('x-acc [m/s^2]');
%     title('Accelerations - PPG'); 
%     subplot(412),plot(t_ppg,accy_ppg), xlabel('Time [s]'), ylabel('y-acc [m/s^2]');
%     subplot(413),plot(t_ppg,accz_ppg), xlabel('Time [s]'), ylabel('z-acc [m/s^2]'); 
%     subplot(414),plot(t_ppg,smv_ppg), xlabel('Time [s]'), ylabel('tot acc [m/s^2]');
%     hold on
%     plot(t_ppg(tempi_task(i,3)),smv_ppg(tempi_task(i,3)),'*k',t_ppg(tempi_task(i,4)),smv_ppg(tempi_task(i,4)),'*k'),
%     hold off
%     figure()
%     subplot(311),plot(t_ppg,gyrx_ppg), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title('Angular velocities - PPG');
%     subplot(312),plot(t_ppg,gyry_ppg), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t_ppg,gyrz_ppg), xlabel('Time [s]'),
end

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
% t=dati_acc{7,2}.S_83B4_ECG_Timestamp_Unix_CAL;
% gyrx_ecg=dati_acc{7,2}.S_83B4_ECG_Gyro_X_CAL;
% gyry_ecg=dati_acc{7,2}.S_83B4_ECG_Gyro_Y_CAL;
% gyrz_ecg=dati_acc{7,2}.S_83B4_ECG_Gyro_Z_CAL;
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
%     figure()
%     subplot(311),plot(t,gyrx_ecg), xlabel('Time [s]'), ylabel('x-ang vel [rad/s]'); 
%     title('Angular velocities - ECG');
%     subplot(312),plot(t,gyry_ecg), xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
%     subplot(313),plot(t,gyrz_ecg), xlabel('Time [s]'),
%     ylabel('z-ang vel [rad/s]');

%% Feature extraction
label = [1,2,3,4,5];

% task 1
finestre_task1=[];
tab1_imu=[]; tab1_ecg=[]; tab1_ppg=[];
for i=1:15
    t=dati{i,1}.imu_Timestamp_Unix_CAL(tempi_task(i,1):tempi_task(i,2));
    gyry_ecg=dati_acc{i,2}.S_83B4_ECG_Gyro_Y_CAL(tempi_task(i,1):tempi_task(i,2));
    [~, peak] = findpeaks(abs(gyry_ecg),'MinPeakHeight',30,'MinPeakDistance',10*FS);
%     figure()
%     plot(t, gyry_ecg, t(peak), gyry_ecg(peak), 'k*');
%     title(['Subject ',num2str(i),' - ECG y-acceleration']);
%     xlabel('Time [s]'), ylabel('y-ang vel [rad/s]');
    for j=1:3
        cont = tempi_task(i,1) + 3*FS;
        k=1;
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        while cont + 8*FS < tempi_task(i,2) - 3*FS | k <= length(peak)
            if cont + 8*FS < peak(k)
                finestre_task1 = [finestre_task1; [cont, cont + 8*FS - 1]];
                tt = t(cont : cont + 8*FS - 1);
                smv = smv_sensor(cont : cont + 8*FS - 1);
                feat1 = features(tt, smv, FS, i, j, label(1));
                if j==1
                tab1_imu = [tab1_imu; feat1];
                elseif j==2 
                tab1_ecg = [tab1_ecg; feat1];
                else
                tab1_ppg = [tab1_ppg; feat1];
                end
                cont = cont+8*FS;
            else
                cont = peak(k) + FS;
                k = k+1;
            end
        end
    end        
end

% task 2
finestre_task2=[];
tab2_imu=[]; tab2_ecg=[]; tab2_ppg=[];
for i=1:n
    ind = [];
    t = dati_acc{i,3}.S_COD4_PPG_Timestamp_Unix_CAL(tempi_task(i,3)+3*FS:tempi_task(i,4)-FS);
    accy_ppg=dati_acc{i,3}.S_COD4_PPG_Accel_WR_Y_CAL(tempi_task(i,3)+3*FS:tempi_task(i,4)-FS);
    ind = task2(t, accy_ppg, i, FS);
    ind = ind + tempi_task(i,3) + 3*FS - 1;
    t = dati_acc{i,3}.S_COD4_PPG_Timestamp_Unix_CAL;   
    for j=1:3
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        for k=2:length(ind)-1
            finestre_task2 = [finestre_task2; [ind(k) - 4*FS , ind(k) + 4*FS - 1]];
            tt = t(ind(k) - 4*FS : ind(k) + 4*FS - 1);
            smv = smv_sensor(ind(k) - 4*FS : ind(k) + 4*FS - 1);
            feat2 = features(tt, smv, FS, i, j, label(2));
            if j==1
                tab2_imu = [tab2_imu; feat2];
            elseif j==2 
                tab2_ecg = [tab2_ecg; feat2];
            else
                tab2_ppg = [tab2_ppg; feat2];
            end
        end
    end
end 

% task 3
finestre_task3=[];
tab3_imu=[]; tab3_ecg=[]; tab3_ppg=[];
for i=1:n
    t=dati_acc{i,1}.imu_Timestamp_Unix_CAL;
    for j=1:3
        cont = tempi_task(i,5) + 3*FS;
        k=1;
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        while cont + 8*FS < tempi_task(i,6) - 3*FS
            finestre_task3 = [finestre_task3; [cont, cont + 8*FS - 1]];
            tt = t(cont : cont + 8*FS - 1);
            smv = smv_sensor(cont : cont + 8*FS - 1);
            feat3 = features(tt, smv, FS, i, j, label(3));
            if j==1
                tab3_imu = [tab3_imu; feat3];
            elseif j==2 
                tab3_ecg = [tab3_ecg; feat3];
            else
                tab3_ppg = [tab3_ppg; feat3];
            end
            cont = cont+8*FS;
        end
    end
end

% task 4
finestre_task4=[];
tab4_imu=[]; tab4_ecg=[]; tab4_ppg=[];
for i=1:n
    t=dati_acc{i,1}.imu_Timestamp_Unix_CAL;
    for j=1:3
        cont = tempi_task(i,7) + 3*FS;
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        while cont + 8*FS < tempi_task(i,8) - 3*FS
            finestre_task4 = [finestre_task4; [cont, cont + 8*FS - 1]];
            tt = t(cont : cont + 8*FS - 1);
            smv = smv_sensor(cont : cont + 8*FS - 1);
            feat4 = features(tt, smv, FS, i, j, label(4));
            if j==1
                tab4_imu = [tab4_imu; feat4];
            elseif j==2 
                tab4_ecg = [tab4_ecg; feat4];
            else
                tab4_ppg = [tab4_ppg; feat4];
            end
            cont = cont+8*FS;
        end
    end
end

% task 5 
finestre_task5=[];
tab5_imu=[]; tab5_ecg=[]; tab5_ppg=[];
for i=1:n
    t=dati_acc{i,2}.S_83B4_ECG_Timestamp_Unix_CAL(tempi_task(i,9)+8*FS:end);
    accx_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_X_CAL(tempi_task(i,9):end);
    accy_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_Y_CAL(tempi_task(i,9):end);
    accz_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_WR_Z_CAL(tempi_task(i,9):end);
    smv_ecg=dati_acc{i,2}.S_83B4_ECG_Accel_Tot(tempi_task(i,9)+8*FS:end);
    ind = task5(t, smv_ecg, i, FS); % figure 8 caso particolare
    ind = ind + tempi_task(i,9) + 8*FS - 1;
    t = dati_acc{i, 2}.S_83B4_ECG_Timestamp_Unix_CAL; 
    for j=1:3
        if j==1
            smv_sensor=dati_acc{i,j}.imu_Accel_Tot;
        elseif j==2 
            smv_sensor=dati_acc{i,j}.S_83B4_ECG_Accel_Tot;
        else
            smv_sensor=dati_acc{i,j}.S_COD4_PPG_Accel_Tot;
        end
        for k=1:length(ind)-1
            finestre_task5 = [finestre_task5; [ind(k) - 4*FS , ind(k) + 4*FS - 1]];
            tt = t(ind(k) - 4*FS : ind(k) + 4*FS - 1);
            smv = smv_sensor(ind(k) - 4*FS : ind(k) + 4*FS - 1);
            feat5 = features(tt, smv, FS, i, j, label(5));
            if j==1
                tab5_imu = [tab5_imu; feat5];
            elseif j==2 
                tab5_ecg = [tab5_ecg; feat5];
            else
                tab5_ppg = [tab5_ppg; feat5];
            end
        end
    end
end

% features = [media, sd, f1, p1, f2, p2, total_power, f625, p625, ratio_p1_tot];

% standardizzazione (da fare su PYTHON) 
tab_imu = [tab1_imu; tab2_imu; tab3_imu; tab4_imu; tab5_imu];
tab_ecg = [tab1_ecg; tab2_ecg; tab3_ecg; tab4_ecg; tab5_ecg];
tab_ppg = [tab1_ppg; tab2_ppg; tab3_ppg; tab4_ppg; tab5_ppg];
tab1 = [tab1_imu(:, 1:end-2), tab1_ecg(:, 1:end-2), tab1_ppg(:, 1:end-1)];
tab2 = [tab2_imu(:, 1:end-2), tab2_ecg(:, 1:end-2), tab2_ppg(:, 1:end-1)];
tab3 = [tab3_imu(:, 1:end-2), tab3_ecg(:, 1:end-2), tab3_ppg(:, 1:end-1)];
tab4 = [tab4_imu(:, 1:end-2), tab4_ecg(:, 1:end-2), tab4_ppg(:, 1:end-1)];
tab5 = [tab5_imu(:, 1:end-2), tab5_ecg(:, 1:end-2), tab5_ppg(:, 1:end-1)];

% % Singolo soggetto (soggetto 2)
% tab_imu_subj2 = tab_imu(find(tab_imu(:,end) == 2), 1:end-1);
% tab_ecg_subj2 = tab_ecg(find(tab_ecg(:,end) == 2), 1:end-1);
% tab_ppg_subj2 = tab_ppg(find(tab_ppg(:,12) == 2), 1:end-1);
% tab_subj2 = [tab_imu_subj2(:, 1:end-1), tab_ecg_subj2(:, 1:end-1), tab_ppg_subj2(:, 1:end)];

% % Generale
% tab_loso = [tab_imu(:, 1:end-2), tab_ecg(:, 1:end-2), tab_ppg];
% tab_tot = [tab1; tab2; tab3; tab4; tab5];
% tab_imu = tab_imu(:, 1:end-1);
% tab_ecg = tab_ecg(:, 1:end-1);
% tab_ppg = tab_ppg(:, 1:end-1);
% tab_imu_ecg = [tab_imu(:, 1:end-1), tab_ecg];
% tab_imu_ppg = [tab_imu(:, 1:end-1), tab_ppg];
% tab_ecg_ppg = [tab_ecg(:, 1:end-1), tab_ppg];
