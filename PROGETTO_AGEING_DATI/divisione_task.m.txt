function tempi_task = divisione_task(dati_imu, FS, subj)
    tempi_task = zeros(10,1);
    tempi_task(1) = 1;
    t = dati_imu.imu_Timestamp_Unix_CAL;
    tempi_task(10) = t(end)*FS;
    acc_x = dati_imu.imu_Accel_WR_X_CAL;
    acc_y = dati_imu.imu_Accel_WR_Y_CAL;
    acc_z = dati_imu.imu_Accel_WR_Z_CAL;
    svm = dati_imu.imu_Accel_Tot;
    [~, tempi_task(2)] = max(svm(111*FS:130*FS));
    tempi_task(2) = tempi_task(2) + 111*FS - 1;
    [~, tempi_task(3)] = max(svm(tempi_task(2)+85*FS : tempi_task(2)+99*FS)); 
    tempi_task(3) = tempi_task(3) + tempi_task(2)+85*FS - 1;
    if subj==12
        [~, tempi_task(4)] = max(svm(tempi_task(3)+190*FS : tempi_task(3)+199*FS));
        tempi_task(4) = tempi_task(4) + tempi_task(3)+190*FS - 1;
    else
        [~, tempi_task(4)] = max(svm(tempi_task(3)+136*FS : tempi_task(3)+158*FS));
        tempi_task(4) = tempi_task(4) + tempi_task(3)+136*FS - 1;
    end
    [~, tempi_task(5)] = max(svm(tempi_task(4)+55*FS : tempi_task(4)+65.5*FS));
    tempi_task(5) = tempi_task(5) + tempi_task(4)+55*FS - 1;
    [~, tempi_task(6)] = max(svm(tempi_task(5)+103*FS : tempi_task(5)+185*FS));
    tempi_task(6) = tempi_task(6) + tempi_task(5)+103*FS - 1;
    [~, tempi_task(7)] = max(svm(tempi_task(6)+113*FS : tempi_task(6)+127*FS)); 
    tempi_task(7) = tempi_task(7) + tempi_task(6)+113*FS - 1;
    [~, tempi_task(8)] = max(abs(svm(tempi_task(7)+55*FS : tempi_task(7)+65.5*FS)));
    tempi_task(8) = tempi_task(8) + tempi_task(7)+55*FS - 1;
    [~, tempi_task(9)] = max(abs(svm(tempi_task(8)+55*FS : tempi_task(8)+65.5*FS)));
    tempi_task(9) = tempi_task(9) + tempi_task(8)+55*FS - 1;
    figure(subj)
    subplot(411), plot(t,acc_x),xlabel('Time [s]'), ylabel('x-acceleration [m/s^2]');
    title(['Subject ',num2str(subj),' - IMU accelerations']);
    subplot(412), plot(t,acc_y), xlabel('Time [s]'), ylabel('y-acceleration [m/s^2]');
    subplot(413), plot(t,acc_z), xlabel('Time [s]'), ylabel('z-acceleration [m/s^2]');
    subplot(414), plot(t,svm), xlabel('Time [s]'), ylabel('z-acceleration [m/s^2]');
    hold on
    plot(t(tempi_task(1)),svm(tempi_task(1)),'*k',t(tempi_task(2)),svm(tempi_task(2)),'*k'),
    plot(t(tempi_task(3)),svm(tempi_task(3)),'*k',t(tempi_task(4)),svm(tempi_task(4)),'*k'),
    plot(t(tempi_task(5)),svm(tempi_task(5)),'*k',t(tempi_task(6)),svm(tempi_task(6)),'*k'),
    plot(t(tempi_task(7)),svm(tempi_task(7)),'*k',t(tempi_task(8)),svm(tempi_task(8)),'*k'),
    plot(t(tempi_task(9)),svm(tempi_task(9)),'*k',t(tempi_task(10)),svm(tempi_task(10)),'*k'),
    hold off
end