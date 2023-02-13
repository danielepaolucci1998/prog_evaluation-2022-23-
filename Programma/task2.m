function ind = task2(t, accy_ppg, subj, FS)
    if subj==7
        [~,locs] = findpeaks(accy_ppg, 'MinPeakDistance', 0.5*FS, 'MinPeakHeight', 1);
    else
        [~,locs] = findpeaks(accy_ppg, 'MinPeakDistance', 0.5*FS, 'MinPeakHeight', 1.8);
    end
    k=1; el=1;
    ind(k) = locs(1);
    for j=2:length(locs)
        if (locs(j)-locs(j-1)) < 4*FS 
            ind(k) = ind(k) + locs(j);
            el = el + 1;
        else
            ind(k) = round(ind(k)/el);
            k = k+1;
            ind(k) = locs(j);
            el = 1;
        end
    end
    ind(end) = round(ind(end)/el);
    % casi particolari
    if subj==3
        ind(2) = [];
    elseif subj==12
        ind(3) = [];
    end
%     figure()
%     plot(t,accy_ppg), xlabel('Time [s]'), ylabel('y-acc [m/s^2]');
%     title(['Subject ',num2str(subj),' - PPG y-acceleration']);
%     hold on
%     plot(t(locs), accy_ppg(locs), 'k*', t(ind), accy_ppg(ind), 'r*');
%     hold off
end