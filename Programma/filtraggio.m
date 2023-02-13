function [dati_imu, dati_ecg, dati_ppg] = filtraggio(dati_imu, dati_ecg, dati_ppg, FS)
    % Band-Pass filter
    [C,D]=butter(4,[0.1,16]/(FS/2)); % Per accelerometro/giroscopio
    % dati IMU
    dati_imu.imu_Accel_WR_X_CAL = filtfilt(C,D,dati_imu.imu_Accel_WR_X_CAL);
    dati_imu.imu_Accel_WR_Y_CAL = filtfilt(C,D,dati_imu.imu_Accel_WR_Y_CAL);
    dati_imu.imu_Accel_WR_Z_CAL = filtfilt(C,D,dati_imu.imu_Accel_WR_Z_CAL);
    dati_imu.imu_Accel_Tot = filtfilt(C,D,dati_imu.imu_Accel_Tot);
    dati_imu.imu_Gyro_X_CAL = filtfilt(C,D,dati_imu.imu_Gyro_X_CAL);
    dati_imu.imu_Gyro_Y_CAL = filtfilt(C,D,dati_imu.imu_Gyro_Y_CAL);
    dati_imu.imu_Gyro_Z_CAL = filtfilt(C,D,dati_imu.imu_Gyro_Z_CAL);
    % dati ECG
    dati_ecg.S_83B4_ECG_Accel_WR_X_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Accel_WR_X_CAL);
    dati_ecg.S_83B4_ECG_Accel_WR_Y_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Accel_WR_Y_CAL);
    dati_ecg.S_83B4_ECG_Accel_WR_Z_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Accel_WR_Z_CAL);
    dati_ecg.S_83B4_ECG_Accel_Tot = filtfilt(C,D,dati_ecg.S_83B4_ECG_Accel_Tot);
    dati_ecg.S_83B4_ECG_ECG_EMG_Status1_CAL = dati_ecg.S_83B4_ECG_ECG_EMG_Status1_CAL;
    dati_ecg.S_83B4_ECG_ECG_EMG_Status2_CAL = dati_ecg.S_83B4_ECG_ECG_EMG_Status2_CAL;
    dati_ecg.S_83B4_ECG_ECG_LA_RA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LA_RA_24BIT_CAL;
    dati_ecg.S_83B4_ECG_ECG_LL_LA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LL_LA_24BIT_CAL;
    dati_ecg.S_83B4_ECG_ECG_LL_RA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LL_RA_24BIT_CAL;
    dati_ecg.S_83B4_ECG_ECG_Vx_RL_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_Vx_RL_24BIT_CAL;
    dati_ecg.S_83B4_ECG_Gyro_X_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Gyro_X_CAL);
    dati_ecg.S_83B4_ECG_Gyro_Y_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Gyro_Y_CAL);
    dati_ecg.S_83B4_ECG_Gyro_Z_CAL = filtfilt(C,D,dati_ecg.S_83B4_ECG_Gyro_Z_CAL);
    % Dati PPG
    dati_ppg.S_COD4_PPG_Accel_WR_X_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Accel_WR_X_CAL);
    dati_ppg.S_COD4_PPG_Accel_WR_Y_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Accel_WR_Y_CAL);
    dati_ppg.S_COD4_PPG_Accel_WR_Z_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Accel_WR_Z_CAL);
    dati_ppg.S_COD4_PPG_Accel_Tot = filtfilt(C,D,dati_ppg.S_COD4_PPG_Accel_Tot);
    dati_ppg.S_COD4_PPG_Gyro_X_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Gyro_X_CAL);
    dati_ppg.S_COD4_PPG_Gyro_Y_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Gyro_Y_CAL);
    dati_ppg.S_COD4_PPG_Gyro_Z_CAL = filtfilt(C,D,dati_ppg.S_COD4_PPG_Gyro_Z_CAL);
    dati_ppg.S_COD4_PPG_PPG_A13_CAL = dati_ppg.S_COD4_PPG_PPG_A13_CAL;

%     % no gravità (g)
%     dati_imu.imu_Accel_Tot = dati_imu.imu_Accel_Tot - 9.81;
%     dati_ecg.S_83B4_ECG_Accel_Tot = dati_ecg.S_83B4_ECG_Accel_Tot - 9.81;
%     dati_ppg.S_COD4_PPG_Accel_Tot = dati_ppg.S_COD4_PPG_Accel_Tot - 9.81;
end