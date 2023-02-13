function feat4 = features(t, smv, FS, subj, j, label)
        media = mean(smv);
        sd = std(smv);
        X = fft(smv);
        N = length(X);
        f = [0:N-1]*FS/N;
        [pxx, freq] = pwelch(smv, [], [], 2*N, 2*FS); % PSD stimata con correlogram, 2N e 2FS per avere stessa risoluzione e stesso numero di campioni (+1) tra f e freq
        indexes = find(f>0.3 & f<15);
        total_power = trapz(pxx(indexes)); % Potenza nel range [0.3 ; 15 Hz]
        [~, ind1] = max(abs(X));
        p1 = pxx(ind1); % Potenza relativa alla frequenza dominante f1
        f1 = f(ind1); % Frequenza dominante in ogni intervallo
        [M, indM] = findpeaks(abs(X), 'NPeaks', 2);
        if M(1) > M(2)
            ind1 = indM(1);
            ind2 = indM(2);
        else
            ind1 = indM(2);
            ind2 = indM(1);
        end
        p1 = pxx(ind1); % Potenza relativa alla frequenza dominante f1
        f1 = f(ind1); % Frequenza dominante in ogni intervallo
        p2 = pxx(ind2); % Potenza relativa alla seconda frequenza dominante f1
        f2 = f(ind2); % Seconda frequenza dominante in ogni intervallo
        if j==1 | j==2
            p2=[];
            f2=[];
        end
        indexes = find(f>0.6 & f<2.5);
        [~, ind625] = max(abs(X(indexes)));
        ind625 = ind625 + indexes(1) - 1;
        p625 = pxx(ind625); % Potenza relativa alla frequenza dominante nel range [0.6 ; 2.5 Hz]
        f625 = f(ind625); % Frequenza dominante nel range [0.6 ; 2.5 Hz]
        ratio_p1_tot = p1/total_power;
        feat4 = [media, sd, f1, p1, f2, p2, total_power, f625, p625, ratio_p1_tot, label, subj];
%         figure()
%         subplot(311), plot(t, smv);
%         xlabel('Time [s]'), ylabel('Acc totale [m/s^2]'), title(['SMV - imu - subj ', num2str(subj), ' task 4']);
%         subplot(312), semilogy(f(round(1:N/2)), (abs(X(round(1:N/2)))), f(ind1), abs(X(ind1)), 'k*', f(ind625), abs(X(ind625)), 'r*');
%         xlabel('Freq [Hz]'), ylabel('Magnitude [-]'), legend('Amplitude Spectrum','Dominant frequency','High Freq in the range [0.6 2.5]');
%         title('Amplitude Spectrum of Acc x - imu - task 4');
%         subplot(313), plot(freq, pxx, freq(ind1), pxx(ind1), 'k*', freq(ind625), pxx(ind625), 'r*')
%         xlabel('Freq'), ylabel('PSD'), legend('Amplitude Spectrum','Dominant frequency','High Freq in the range [0.6 2.5]'); 
%         title('Power Spectral Density');
end