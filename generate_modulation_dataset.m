%% ===============================
%  Complex Baseband Modulation Generator
%  Production-Ready for CNN Dataset
%  Author: Thabiso
%% ===============================

clc; clear; close all;

%% Parameters
fs = 1000;                % Sampling frequency (Hz)
N = 1024;                 % Samples per frame
t = (0:N-1)/fs;           % Time vector
numSymbols = 32;          % Number of symbols for digital modulations
sps = N/numSymbols;       % Samples per symbol
m = 0.5;                  % AM modulation index
freqDev = 5;              % FM frequency deviation (Hz)
fc = 0;                   % Carrier frequency for AM (set to 0 for baseband)
freqOffset = 0.05;        % Normalized frequency offset
phaseOffset = pi/4;       % Phase offset
numFrames = 1000;         % Number of frames per modulation
SNR_range = [-10 20];     % SNR range for AWGN (dB)

%% Modulation types
modTypes = {'AM','FM','PM','BPSK','QPSK','BFSK'};
numMods = length(modTypes);

%% Preallocate storage
allSignals = zeros(numFrames, N, numMods);

%% Main generation loop
for idxMod = 1:numMods
    for frame = 1:numFrames
        %% Generate smooth analog/digital signals
        switch modTypes{idxMod}
            case 'AM'
                message = sin(2*pi*10*t);               % Smooth 10 Hz tone
                x = (1 + m*message) .* exp(1j*2*pi*fc*t);

            case 'FM'
                message = sin(2*pi*5*t);                % Smooth 5 Hz tone
                phase = 2*pi*freqDev * cumsum(message)/fs;
                x = exp(1j*phase);

            case 'PM'
                message = sin(2*pi*5*t);                % Smooth 5 Hz tone
                x = exp(1j*pi*message);

            case 'BPSK'
                bits = randi([0 1], 1, numSymbols);
                symbols = 2*bits - 1;                   % Map 0->-1, 1->1
                x = repelem(symbols, sps);
                x = exp(1j*pi*(x==1));                  % Phase 0 or pi

            case 'QPSK'
                bits = randi([0 1], 1, 2*numSymbols);  % 2 bits per symbol
                symbolMap = bits(1:2:end)*2 + bits(2:2:end); 
                phases = [0 pi/2 pi 3*pi/2];
                x = repelem(phases(symbolMap+1), sps);
                x = exp(1j*x);

            case 'BFSK'
                bits = randi([0 1], 1, numSymbols);
                f1 = 0.1; f2 = 0.2;                     % Normalized frequencies
                freqs = (bits==0)*f1 + (bits==1)*f2;
                inst_freq = repelem(freqs, sps);
                phase = 2*pi * cumsum(inst_freq)/fs;
                x = exp(1j*phase);
        end

        %% Apply Rayleigh Fading (constant per frame)
        h = (randn + 1j*randn)/sqrt(2);
        x = x * h;

        %% Apply frequency and phase offsets
        x = x .* exp(1j*(2*pi*freqOffset*t + phaseOffset));

        %% Normalize power to unit energy (crucial for CNN)
        x = x / sqrt(mean(abs(x).^2));

        %% Add random AWGN for robustness
        SNR_dB = randi(SNR_range);
        x = awgn(x, SNR_dB, 'measured');

        %% Store signal
        allSignals(frame,:,idxMod) = x;
    end
end

%% Save dataset
save('mod_data.mat', 'allSignals', 'modTypes', 'fs');

%% Example Plot
figure;
for idxMod = 1:numMods
    subplot(3,2,idxMod)
    plot(real(allSignals(1,:,idxMod))); hold on;
    plot(imag(allSignals(1,:,idxMod)));
    title(modTypes{idxMod});
    xlabel('Sample'); ylabel('Amplitude');
    legend('I','Q')
end
sgtitle('Normalized Complex Baseband Signals with Noise')