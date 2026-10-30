function [H, pathsSelected] = ckmsrf_synthesize_history_channel( ...
    scene, positionXYZ, ueYawRad, cfg, sampleSeed)
%CKMSRF_SYNTHESIZE_HISTORY_CHANNEL One narrowband historical realization.

rng(sampleSeed, 'twister');
paths = ckmsrf_compute_paths(scene, positionXYZ, ueYawRad, cfg);
[~, order] = sort(paths.powerLin, 'descend');
order = order(1:cfg.numPaths);
phase = 2*pi*rand(numel(paths.pathId), 1);

H = complex(zeros(cfg.numRxAntennas, cfg.numTxAntennas), ...
            zeros(cfg.numRxAntennas, cfg.numTxAntennas));
for l = 1:numel(order)
    idx = order(l);
    if ~paths.mask(idx) || paths.powerLin(idx) <= 0
        continue;
    end
    gain = paths.amplitude(idx) * exp(1j * ( ...
        phase(idx) - 2*pi*cfg.carrierFrequencyHz*paths.delaySec(idx)));
    at = ckmsrf_upa_steering( ...
        cfg.txArraySize, paths.aodAzRad(idx), paths.aodElRad(idx));
    ar = ckmsrf_upa_steering( ...
        cfg.rxArraySize, paths.aoaAzRad(idx), paths.aoaElRad(idx));
    H = H + gain * (ar * at');
end

fields = fieldnames(paths);
pathsSelected = struct();
for i = 1:numel(fields)
    value = paths.(fields{i});
    if size(value, 1) == numel(paths.pathId)
        pathsSelected.(fields{i}) = value(order, :);
    end
end

end
