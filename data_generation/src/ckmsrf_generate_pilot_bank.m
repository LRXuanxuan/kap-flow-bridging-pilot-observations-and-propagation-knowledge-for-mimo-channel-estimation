function bank = ckmsrf_generate_pilot_bank(cfg, sceneId)
%CKMSRF_GENERATE_PILOT_BANK Reproducible semi-unitary DFT pilot matrices.

Nt = cfg.numTxAntennas;
Np = cfg.pilot.numPilots;
B = cfg.pilot.numPilotMatrices;
F = localDft(Nt);
P = complex(zeros(Nt, Np, B, 'single'), zeros(Nt, Np, B, 'single'));
selectedIndex = zeros(Np, B, 'uint16');

rng(cfg.randomSeed + 400000 * sceneId, 'twister');
for b = 1:B
    index = randperm(Nt, Np);
    P(:, :, b) = single(F(:, index));
    selectedIndex(:, b) = uint16(index(:));
end

bank = struct();
bank.P = P;
bank.selectedDftColumn = selectedIndex;
bank.numPilots = uint16(Np);
bank.numTransmitAntennas = uint16(Nt);
bank.density = single(Np / Nt);
bank.seed = uint32(cfg.randomSeed + 400000 * sceneId);

end

function F = localDft(N)
n = (0:N-1).';
k = 0:N-1;
F = exp(-1j * 2*pi * n * k / N) / sqrt(N);
end
