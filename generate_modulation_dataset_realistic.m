%% ===============================
%  Complex Baseband Modulation Generator
%  V3: HIGH-REALISM UPGRADE
%  Author: Thabiso (Final Research Grade)
%% ===============================

clc; clear; close all;

%% Parameters
fs = 1e6;                 
N = 1024;                 
numSymbols = 128;         
sps = 8;                  
numFrames = 2000;         % INCREASED: More data for higher accuracy
SNR_range = [-10 30];     

modTypes = {'AM','FM','PM','BPSK','QPSK','BFSK'};
numMods = length(modTypes);

% --- Pulse Shaper Setup ---
rolloff = 0.35; 
span = 6; % Increased span for better pulse shape definition
h_filter = rcosdesign(rolloff, span, sps);

%% Preallocate
allSignals = zeros(numFrames, N, numMods);

%% Main loop
for idxMod = 1:numMods
    fprintf('Generating %s...\n', modTypes{idxMod});
    for frame = 1:numFrames

        SNR_dB = randi(SNR_range);
        
        % 1. IMPROVED OFFSETS: Randomized per frame
        freqOffset = (rand()*0.02 - 0.01); % Max 1% freq shift
        phaseOffset = rand()*2*pi;      
        
        % 2. SAMPLING DRIFT: Mimics real hardware clock error
        clockDrift = 1 + (rand()*0.001 - 0.0005); % ±0.05% drift
        t = (0:N-1) * clockDrift; 

        %% ===============================
        % SIGNAL GENERATION (CORE)
        %% ===============================
        switch modTypes{idxMod}
            case 'AM'
                msg = resample(randn(1, numSymbols), N, numSymbols);
                m = 0.5 + 0.3*rand(); 
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
                x_pulse = upfirdn(syms, h_filter, sps);
                x = x_pulse(1:N);
            case 'QPSK'
                bits = randi([0 1], 1, 2*numSymbols);
                syms = pskmod(bits.', 4, pi/4, 'InputType', 'bit').';
                x_pulse = upfirdn(syms, h_filter, sps);
                x = x_pulse(1:N);
            case 'BFSK'
                bits = randi([0 1], 1, numSymbols);
                x = fskmod(bits, 2, 0.1*fs, sps, fs);
                x = x(1:N);
        end

        %% ===============================
        % UPGRADED CHANNEL EFFECTS
        %% ===============================
        % 1. RICIAN FADING (More realistic than just Rayleigh)
        % This adds a direct Line-of-Sight component + scattered multipath
        K = 4; % Rician K-factor
        los = sqrt(K/(K+1)); 
        scatter = sqrt(1/(K+1)) * (randn(1,N) + 1j*randn(1,N))/sqrt(2);
        h = filter(ones(1,5)/5, 1, los + scatter); 
        x = x .* h;

        % 2. Apply Freq/Phase/Timing offsets
        x = x .* exp(1j*(2*pi*freqOffset*t + phaseOffset));

        % 3. Power Normalization
        x = x / sqrt(mean(abs(x).^2));

        % 4. Add AWGN
        x = awgn(x, SNR_dB, 'measured');

        allSignals(frame,:,idxMod) = x;
    end
end

save('mod_data.mat', 'allSignals', 'modTypes', 'fs');
fprintf('Success! V3 Dataset generated.\n');
