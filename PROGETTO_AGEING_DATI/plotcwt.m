function plotcwt(signal,fps)
 time_t = [1:length(signal)];
    [cfs_ppg,freq_c_ppg] = cwt(signal,"bump",fps);
    tms = ((0:numel(time_t)-1)/fps);
    
    figure
    subplot(2,1,1)
    plot(tms,signal)
    title("Signal and Scalogram")
    xlabel("Time (s)")
    ylabel("Amplitude")
    subplot(2,1,2)
    surface(tms,freq_c_ppg,abs(cfs_ppg));
    axis tight
    shading flat
    colormap("default")
    xlabel("Time (s)")
    ylabel("Frequency (Hz)")
    set(gca,"yscale","log")
end