function [Y, noiseVar] = ckmsrf_add_awgn(Y0, snrDb, sampleSeed)
%CKMSRF_ADD_AWGN Add complex AWGN at the requested measured SNR.

rng(sampleSeed, 'twister');
signalPower = mean(abs(Y0(:)).^2);
noiseVar = max(signalPower / (10^(snrDb/10)), realmin);
noise = sqrt(noiseVar/2) * ...
    (randn(size(Y0)) + 1j*randn(size(Y0)));
Y = Y0 + noise;

end
