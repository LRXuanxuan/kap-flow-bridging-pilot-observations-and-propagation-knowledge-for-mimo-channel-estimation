function cfg = ckmsrf_validate_config(cfg)
%CKMSRF_VALIDATE_CONFIG Validate and finalize derived configuration fields.

if prod(cfg.txArraySize) ~= cfg.numTxAntennas
    error('CKMSRF:Config', 'txArraySize must multiply to numTxAntennas.');
end
if prod(cfg.rxArraySize) ~= cfg.numRxAntennas
    error('CKMSRF:Config', 'rxArraySize must multiply to numRxAntennas.');
end
if cfg.numPaths > cfg.numScatterers + 1
    error('CKMSRF:Config', 'numPaths cannot exceed numScatterers + one LoS path.');
end
if cfg.areaX(1) >= cfg.areaX(2) || cfg.areaY(1) >= cfg.areaY(2)
    error('CKMSRF:Config', 'Invalid areaX or areaY.');
end
if cfg.numSubcarriers < 1 || cfg.numSnapshots < 1
    error('CKMSRF:Config', 'numSubcarriers and numSnapshots must be positive.');
end
if cfg.ckm.gridResolutionM <= 0 || cfg.split.blockSizeM <= 0
    error('CKMSRF:Config', 'CKM grid and split block sizes must be positive.');
end

splitSum = cfg.split.trainFraction + cfg.split.valFraction + cfg.split.testFraction;
if abs(splitSum - 1) > 1e-9
    error('CKMSRF:Config', 'Spatial split fractions must sum to one.');
end

if isempty(cfg.pilot.numPilots)
    cfg.pilot.numPilots = max(1, round(cfg.pilot.density * cfg.numTxAntennas));
end
if cfg.pilot.numPilots < 1 || cfg.pilot.numPilots > cfg.numTxAntennas
    error('CKMSRF:Config', 'numPilots must lie in [1, numTxAntennas].');
end
cfg.pilot.density = cfg.pilot.numPilots / cfg.numTxAntennas;

if cfg.dataset.shardSize < 1
    error('CKMSRF:Config', 'dataset.shardSize must be positive.');
end
if cfg.ckm.historySamplesPerCell < 1
    error('CKMSRF:Config', 'historySamplesPerCell must be positive.');
end
if ~ismember(lower(cfg.ueOrientationMode), {'heading', 'fixed', 'random'})
    error('CKMSRF:Config', 'Unsupported ueOrientationMode.');
end
if cfg.localization.errorStdM < 0
    error('CKMSRF:Config', 'localization.errorStdM cannot be negative.');
end

cfg.subcarrierOffsetHz = ((0:cfg.numSubcarriers-1) ...
    - (cfg.numSubcarriers-1)/2) * cfg.subcarrierSpacingHz;
cfg.timeAxisSec = (0:cfg.numSnapshots-1) * cfg.snapshotPeriodSec;

end
