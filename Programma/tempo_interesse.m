function [dati_imu, dati_ecg, dati_ppg] = tempo_interesse(dati_imu, dati_ecg, dati_ppg, inizio, fine);
    % dati IMU
    dati_imu.imu_Timestamp_Unix_CAL = datetime(dati_imu.imu_Timestamp_Unix_CAL/1000,'convertfrom','posixtime','timezone','Europe/Rome');
    indmin = find(dati_imu.imu_Timestamp_Unix_CAL == inizio);
    indmax = find(dati_imu.imu_Timestamp_Unix_CAL == fine);
    interval = indmax - indmin;
    dati_imu.imu_Timestamp_Unix_CAL = dati_imu.imu_Timestamp_Unix_CAL(indmin:indmax);
    dati_imu.imu_Accel_WR_X_CAL = dati_imu.imu_Accel_WR_X_CAL(indmin:indmax);
    dati_imu.imu_Accel_WR_Y_CAL = dati_imu.imu_Accel_WR_Y_CAL(indmin:indmax);
    dati_imu.imu_Accel_WR_Z_CAL = dati_imu.imu_Accel_WR_Z_CAL(indmin:indmax);
    dati_imu.imu_Accel_Tot = sqrt(dati_imu.imu_Accel_WR_X_CAL.^2 + dati_imu.imu_Accel_WR_Y_CAL.^2 + dati_imu.imu_Accel_WR_Z_CAL.^2);
    dati_imu.imu_Gyro_X_CAL = dati_imu.imu_Gyro_X_CAL(indmin:indmax);
    dati_imu.imu_Gyro_Y_CAL = dati_imu.imu_Gyro_Y_CAL(indmin:indmax);
    dati_imu.imu_Gyro_Z_CAL = dati_imu.imu_Gyro_Z_CAL(indmin:indmax);
    % dati ECG
    dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL = datetime(dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL/1000,'convertfrom','posixtime','timezone','Europe/Rome');
    indmin = find(dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL <= inizio); 
    indmin = indmin(end);
    indmax = indmin + interval;
    if indmax > length(dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL)
        indmax = length(dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL);
    end
    dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL = dati_ecg.S_83B4_ECG_Timestamp_Unix_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Accel_WR_X_CAL = dati_ecg.S_83B4_ECG_Accel_WR_X_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Accel_WR_Y_CAL = dati_ecg.S_83B4_ECG_Accel_WR_Y_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Accel_WR_Z_CAL = dati_ecg.S_83B4_ECG_Accel_WR_Z_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Accel_Tot = sqrt(dati_ecg.S_83B4_ECG_Accel_WR_X_CAL.^2 + dati_ecg.S_83B4_ECG_Accel_WR_Y_CAL.^2 + dati_ecg.S_83B4_ECG_Accel_WR_Z_CAL.^2);
    dati_ecg.S_83B4_ECG_ECG_EMG_Status1_CAL = dati_ecg.S_83B4_ECG_ECG_EMG_Status1_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_ECG_EMG_Status2_CAL = dati_ecg.S_83B4_ECG_ECG_EMG_Status2_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_ECG_LA_RA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LA_RA_24BIT_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_ECG_LL_LA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LL_LA_24BIT_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_ECG_LL_RA_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_LL_RA_24BIT_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_ECG_Vx_RL_24BIT_CAL = dati_ecg.S_83B4_ECG_ECG_Vx_RL_24BIT_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Gyro_X_CAL = dati_ecg.S_83B4_ECG_Gyro_X_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Gyro_Y_CAL = dati_ecg.S_83B4_ECG_Gyro_Y_CAL(indmin:indmax);
    dati_ecg.S_83B4_ECG_Gyro_Z_CAL = dati_ecg.S_83B4_ECG_Gyro_Z_CAL(indmin:indmax);
    % Dati PPG
    dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL = datetime(dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL/1000,'convertfrom','posixtime','timezone','Europe/Rome');
    indmin = find(dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL <= inizio); 
    indmin = indmin(end);
    indmax = indmin + interval;
    if indmax > length(dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL)
        indmax = length(dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL);
    end
    dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL = dati_ppg.S_COD4_PPG_Timestamp_Unix_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Accel_WR_X_CAL = dati_ppg.S_COD4_PPG_Accel_WR_X_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Accel_WR_Y_CAL = dati_ppg.S_COD4_PPG_Accel_WR_Y_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Accel_WR_Z_CAL = dati_ppg.S_COD4_PPG_Accel_WR_Z_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Accel_Tot = sqrt(dati_ppg.S_COD4_PPG_Accel_WR_X_CAL.^2 + dati_ppg.S_COD4_PPG_Accel_WR_Y_CAL.^2 + dati_ppg.S_COD4_PPG_Accel_WR_Z_CAL.^2);
    dati_ppg.S_COD4_PPG_Gyro_X_CAL = dati_ppg.S_COD4_PPG_Gyro_X_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Gyro_Y_CAL = dati_ppg.S_COD4_PPG_Gyro_Y_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_Gyro_Z_CAL = dati_ppg.S_COD4_PPG_Gyro_Z_CAL(indmin:indmax);
    dati_ppg.S_COD4_PPG_PPG_A13_CAL = dati_ppg.S_COD4_PPG_PPG_A13_CAL(indmin:indmax);










end