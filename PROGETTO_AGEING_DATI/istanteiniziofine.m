function [date_min, date_max] = istanteiniziofine(dati_imu, dati_ecg, dati_ppg, FS, subj)
%     t_imu=(dati_imu.imu_Timestamp_Unix_CAL - dati_imu.imu_Timestamp_Unix_CAL(1))/1000;
    t_imu=dati_imu.imu_Timestamp_Unix_CAL/1000;
    svm_imu = sqrt(dati_imu.imu_Accel_WR_X_CAL.^2 + dati_imu.imu_Accel_WR_Y_CAL.^2 + dati_imu.imu_Accel_WR_Z_CAL.^2);
    [~, ind]=findpeaks(abs(svm_imu),"MinPeakHeight",27,"MinPeakDistance",40*FS);
    indmin=ind(1);
    date_min=datetime(t_imu(indmin),'convertfrom','posixtime','timezone','Europe/Rome');
    indmax=ind(end);
    if subj==12
        indmax = length(dati_imu.imu_Timestamp_Unix_CAL/1000);
    end
    date_max=datetime(t_imu(indmax),'convertfrom','posixtime','timezone','Europe/Rome');
end