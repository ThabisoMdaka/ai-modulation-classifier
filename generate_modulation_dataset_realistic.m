%% ===============================
%  Complex Baseband Modulation Generator
%  PRODUCTION VERSION (Fixes Accuracy Issues)
%  Author: Thabiso (Final Upgrade)
%% ===============================

clc; clear; close all;

%% Parameters
fs = 1e6;                 % Increased to 1MHz for realistic bandwidth
N = 1024;                 % Samples per frame
numSymbols = 128;         % Increased for better pattern recognition
sps = 8;                  % Samples per symbol
numFrames = 1500;         % More data = better generalization

% SNR Floor raised to -10dB. -20dB is pure noise; 
% we need the model to learn the shape before it learns the noise.
SNR_range = [-10 30];     

%% Modulation types
modTypes = {'AM','FM','PM','BPSK','QPSK','BFSK'};
numMods = length(modTypes);

% --- Pulse Shaper Setup ---
% This is the "secret sauce" for real-world signal behavior
rolloff = 0.35; 
span = 4;
h_filter = rcosdesign(rolloff, span, sps);

%% Preallocate
allSignals = zeros(numFrames, N, numMods);

%% Main loop
for idxMod = 1:numMods
    fprintf('Generating %s...\n', modTypes{idxMod});
    for frame = 1:numFrames

        SNR_dB = randi(SNR_range);
        freqOffset = (rand()*0.05 - 0.025); 
        phaseOffset = (rand()*2*pi - pi);      
        t = (0:N-1)/fs;

        %% ===============================
        % SIGNAL GENERATION
        %% ===============================
        switch modTypes{idxMod}

            case 'AM'
                msg = resample(randn(1, numSymbols), N, numSymbols);
                m = 0.5 + 0.5*rand(); 
                x = (1 + m*msg(1:N));

            case 'FM'
                msg = filter(ones(1,10)/10, 1, randn(1, N));
                x = exp(1j*2*pi*0.1*cumsum(msg));

            case 'PM'
                msg = resample(randn(1, numSymbols), N, numSymbols);
                x = exp(1j*pi*msg(1:N));

            case 'BPSK'
                bits = randi([0 1], 1, numSymbols);
                syms = 2*bits - 1;
                % Apply Pulse Shaping
                x_pulse = upfirdn(syms, h_filter, sps);
                x = x_pulse(1:N);

            case 'QPSK'
                bits = randi([0 1], 1, 2*numSymbols);
                syms = pskmod(bits.', 4, pi/4, 'InputType', 'bit').';
                % Apply Pulse Shaping
                x_pulse = upfirdn(syms, h_filter, sps);
                x = x_pulse(1:N);

            case 'BFSK'
                bits = randi([0 1], 1, numSymbols);
                % FSK modulation with realistic frequency separation
                x = fskmod(bits, 2, 0.1*fs, sps, fs);
                x = x(1:N);
        end

        %% ===============================
        % REAL-WORLD CHANNEL EFFECTS
        %% ===============================
        % 1. Rayleigh Fading (Multipath)
        h = (randn(1,N) + 1j*randn(1,N))/sqrt(2);
        h = filter(ones(1,5)/5, 1, h); 
        x = x .* h;

        % 2. Frequency & Phase Offsets
        x = x .* exp(1j*(2*pi*freqOffset*(1:N) + phaseOffset));

        % 3. Power Normalization (Critical for CNN)
        x = x / sqrt(mean(abs(x).^2));

        % 4. Add AWGN
        x = awgn(x, SNR_dB, 'measured');

        allSignals(frame,:,idxMod) = x;
    end
end

save('mod_data.mat', 'allSignals', 'modTypes', 'fs');
fprintf('Success! Dataset generated.\n');

%% VISUALIZATION
figure;
for idxMod = 1:numMods
    subplot(3,2,idxMod);
    plot(real(allSignals(1,1:200,idxMod))); hold on;
    plot(imag(allSignals(1,1:200,idxMod)));
    title(modTypes{idxMod});
    legend('I','Q');
end