% PROGETTO PROVA

clear all
close all
clc

%%
%% PREPROCESSING
%%

% IMU on the ankle, ECG on the chest, PPG on the wrist

subject=["ALESSANDRA","ALEX","CHIARA","CLAUDIA","DANIELE_PA","DANI_G","ELENA","FEDERICA","FRANCESCO","GAETANO","GIORGIA","LUCA","MASSIMO","MATTEO_F","MATTEO_G","SOPHIE"];

Pat = 'Select a Subject : ';
subject = input(Pat);

% subject=["DANIELE_PA"];

sensor=["IMU","ECG","PPG"];

n = length(subject);
L = length(subject)
for i = subject
    for j=1:3
        stringa=strcat(sensor(j),"_",subject(i),'.mat');
        dati{i,j}=importdata(stringa);
    end
end

for i=1:size(dati,1)
    imu_t=dati{i,1}.imu_Timestamp_Unix_CAL;
    ecg_t=dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL;
    ppg_t=dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL;
    t_min(i)=min([size(imu_t,1),size(ecg_t,1),size(ppg_t,1)]);
end
FS=128;

% dati_imu=dati{11,1};
% dati_ecg=dati{11,2};
% dati_ppg=dati{11,3};
% t_imu_mat=(dati_imu.imu_Timestamp_Unix_CAL - dati_imu.imu_Timestamp_Unix_CAL(1))/1000;
% t_ecg_mat=(dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL - dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL(1))/1000;
% t_ppg_mat=(dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL - dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL(1))/1000;

tempi_task = zeros(11,10);
for i=1:L
    [date_min(i), date_max(i)] = istanteiniziofine(dati{i,1}, dati{i,2}, dati{i,3},FS,i);
    [dati{i,1}, dati{i,2}, dati{i,3}] = tempo_interesse(dati{i,1}, dati{i,2}, dati{i,3}, date_min(i), date_max(i));
    dati{i,1}.imu_Timestamp_Unix_CAL = [0:length(dati{i,1}.imu_Timestamp_Unix_CAL)-1]/FS;
    dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL = [0:length(dati{i,2}.S_83B4_ECG_Timestamp_Unix_CAL)-1]/FS;
    dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL = [0:length(dati{i,3}.S_COD4_PPG_Timestamp_Unix_CAL)-1]/FS;
    tempi_task(i,:) = divisione_task(dati{i,1}, FS, i); % casi particolari per Subject 1,4,5
end

fields={'S_83B4_ECG_ECG_EMG_Status1_CAL', 'S_83B4_ECG_ECG_EMG_Status2_CAL', 'S_83B4_ECG_ECG_LA_RA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_LA_24BIT_CAL', 'S_83B4_ECG_ECG_LL_RA_24BIT_CAL', 'S_83B4_ECG_ECG_Vx_RL_24BIT_CAL'};
S=length(fields)
for i=1:S
    dati_acc{i,1}=dati{i,1};
    dati_acc{i,2}=rmfield(dati{i,2},fields);
    dati_acc{i,3}=rmfield(dati{i,3},"S_COD4_PPG_PPG_A13_CAL");
end