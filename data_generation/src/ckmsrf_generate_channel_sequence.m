function sequence = ckmsrf_generate_channel_sequence(scene, trajectory, cfg, sampleSeed)
%CKMSRF_GENERATE_CHANNEL_SEQUENCE Generate a coherent wideband MIMO sequence.
%
% A fixed set of physical path IDs is selected for the entire trajectory.
% The propagation phase is computed from the time-varying geometric delay;
% Doppler is stored as a label and is not added a second time.

rng(sampleSeed, 'twister');
T = cfg.numSnapshots;
Nsc = cfg.numSubcarriers;
Nr = cfg.numRxAntennas;
Nt = cfg.numTxAntennas;
L = cfg.numPaths;
numPhysicalPaths = cfg.numScatterers + 1;

allPaths = cell(T, 1);
averagePower = zeros(numPhysicalPaths, 1);
for t = 1:T
    yaw = deg2rad(trajectory.ueOrientationDeg(t));
    allPaths{t} = ckmsrf_compute_paths( ...
        scene, trajectory.positionXYZ(t, :), yaw, cfg);
    averagePower = averagePower + allPaths{t}.powerLin;
end
averagePower = averagePower / T;
[~, order] = sort(averagePower, 'descend');
selectedPhysicalIndex = order(1:L);
selectedPathId = int32(selectedPhysicalIndex - 1);

initialPhase = 2*pi*rand(numPhysicalPaths, 1);
H = complex(zeros(Nr, Nt, Nsc, T, 'single'), ...
            zeros(Nr, Nt, Nsc, T, 'single'));

truth = struct();
truth.pathId = repmat(selectedPathId, 1, T);
truth.pathMask = false(L, T);
truth.pathGain = complex(zeros(L, T, 'single'), zeros(L, T, 'single'));
truth.pathPowerLin = zeros(L, T, 'single');
truth.delaySec = zeros(L, T, 'single');
truth.aodAzRad = zeros(L, T, 'single');
truth.aodElRad = zeros(L, T, 'single');
truth.aoaAzRad = zeros(L, T, 'single');
truth.aoaElRad = zeros(L, T, 'single');
truth.dopplerHz = zeros(L, T, 'single');

for t = 1:T
    paths = allPaths{t};
    Ht = complex(zeros(Nr, Nt, Nsc), zeros(Nr, Nt, Nsc));
    velocity = trajectory.velocityXYZ(t, :);

    for l = 1:L
        pidx = selectedPhysicalIndex(l);
        delay = paths.delaySec(pidx);
        power = paths.powerLin(pidx);
        mask = paths.mask(pidx) && power > 0;
        fd = (cfg.carrierFrequencyHz / cfg.lightSpeed) * ...
            dot(velocity, paths.arrivalUnitTowardSource(pidx, :));

        truth.pathMask(l, t) = mask;
        truth.pathPowerLin(l, t) = single(power);
        truth.delaySec(l, t) = single(delay);
        truth.aodAzRad(l, t) = single(paths.aodAzRad(pidx));
        truth.aodElRad(l, t) = single(paths.aodElRad(pidx));
        truth.aoaAzRad(l, t) = single(paths.aoaAzRad(pidx));
        truth.aoaElRad(l, t) = single(paths.aoaElRad(pidx));
        truth.dopplerHz(l, t) = single(fd);

        if ~mask
            continue;
        end

        gainCarrier = paths.amplitude(pidx) * exp(1j * ( ...
            initialPhase(pidx) - 2*pi*cfg.carrierFrequencyHz*delay));
        truth.pathGain(l, t) = single(gainCarrier);

        at = ckmsrf_upa_steering( ...
            cfg.txArraySize, paths.aodAzRad(pidx), paths.aodElRad(pidx));
        ar = ckmsrf_upa_steering( ...
            cfg.rxArraySize, paths.aoaAzRad(pidx), paths.aoaElRad(pidx));
        spatialOuter = ar * at';
        frequencyPhase = exp(-1j * 2*pi * cfg.subcarrierOffsetHz * delay);
        for k = 1:Nsc
            Ht(:, :, k) = Ht(:, :, k) ...
                + gainCarrier * frequencyPhase(k) * spatialOuter;
        end
    end
    H(:, :, :, t) = single(Ht);
end

scaleLin = zeros(Nsc, T, 'single');
for t = 1:T
    for k = 1:Nsc
        scaleLin(k, t) = single(norm(double(H(:, :, k, t)), 'fro'));
    end
end

sequence = struct();
sequence.H = H;
sequence.scaleLin = scaleLin;
sequence.truth = truth;
sequence.selectedPathId = selectedPathId;

end
