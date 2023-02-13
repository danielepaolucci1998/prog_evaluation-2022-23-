function ind = task5(t, smv_ecg, subj, FS)
    [~,locs] = findpeaks((smv_ecg), 'MinPeakDistance', 1.5*FS, 'MinPeakHeight', 1.2);
    k=1;
    ind=[];
    if subj==8
        locs(1)=[];
    end
    for j=1:round((length(locs)-1)/2)
        ind(k)=round((locs(2*j-1)+locs(2*j))/2);
        k=k+1;        
    end
%     figure()
%     plot(t,smv_ecg), xlabel('Time [s]'), ylabel('tot-acc [m/s^2]');
%     title(['Subject ',num2str(subj),' - ECG total-acceleration']);
%     hold on
%     plot(t(locs), smv_ecg(locs), 'k*', t(ind), smv_ecg(ind), 'r*');
%     hold off
end



