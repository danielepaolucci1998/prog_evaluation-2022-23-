function [walking,drinking,stairs,sleeping,situp] = task_generator(ecg_signal,tempi_task)
%split the signal according to the task performed
% the input must be a signal filtered that already got the pre-processing procedure. the second input has to be
% the vector of index of start and end task.

    walking=ecg_signal(tempi_task(1):tempi_task(2));
    drinking=ecg_signal(tempi_task(3):tempi_task(4));
    stairs=ecg_signal(tempi_task(5):tempi_task(6));
    sleeping=ecg_signal(tempi_task(7):tempi_task(8));
    situp=ecg_signal(tempi_task(9):tempi_task(10));

end