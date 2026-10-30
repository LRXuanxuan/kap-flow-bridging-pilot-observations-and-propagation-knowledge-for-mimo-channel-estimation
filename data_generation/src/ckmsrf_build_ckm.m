function ckm = ckmsrf_build_ckm(scene, cfg, Ftx, Wrx)
%CKMSRF_BUILD_CKM Build a CKM from independent historical channel samples.
%
% The map contains only aggregated historical statistics. It never stores
% or queries an online test sample's instantaneous path gain.

xGrid = cfg.areaX(1):cfg.ckm.gridResolutionM:cfg.areaX(2);
yGrid = cfg.areaY(1):cfg.ckm.gridResolutionM:cfg.areaY(2);
if xGrid(end) < cfg.areaX(2)
    xGrid(end+1) = cfg.areaX(2);
end
if yGrid(end) < cfg.areaY(2)
    yGrid(end+1) = cfg.areaY(2);
end
[X, Y] = meshgrid(xGrid, yGrid);
gridXY = [X(:) Y(:)];
G = size(gridXY, 1);

nAod = cfg.ckm.numAodBins;
nAoa = cfg.ckm.numAoaBins;
nDelay = cfg.ckm.numDelayBins;
adpDim = nAod * nAoa * nDelay;

% Scalar feature order is documented and stored in the map.
% [LoS probability, pathloss mean dB, pathloss std dB,
%  mean delay (s), RMS delay spread (s), historical sample count].
scalar = zeros(6, G, 'single');
adp = zeros(adpDim, G, 'single');
txBeamProbability = zeros(cfg.numTxAntennas, G, 'single');
rxBeamProbability = zeros(cfg.numRxAntennas, G, 'single');

halfCell = cfg.ckm.gridResolutionM / 2;
R = cfg.ckm.historySamplesPerCell;

for g = 1:G
    losValue = zeros(R, 1);
    pathlossDb = zeros(R, 1);
    meanDelay = zeros(R, 1);
    rmsDelay = zeros(R, 1);
    adpAccum = zeros(adpDim, 1);
    txAccum = zeros(cfg.numTxAntennas, 1);
    rxAccum = zeros(cfg.numRxAntennas, 1);

    for r = 1:R
        seed = cfg.randomSeed + 300000 * double(scene.sceneId) + 1000*g + r;
        rng(seed, 'twister');
        q = gridXY(g, :) + (2*rand(1,2)-1) * halfCell;
        q(1) = min(max(q(1), cfg.areaX(1)), cfg.areaX(2));
        q(2) = min(max(q(2), cfg.areaY(1)), cfg.areaY(2));
        yawRad = localHistoricalYaw(cfg);
        positionXYZ = [q cfg.ueHeightM];

        [H, paths] = ckmsrf_synthesize_history_channel( ...
            scene, positionXYZ, yawRad, cfg, seed + 50000000);
        beam = ckmsrf_beam_features( ...
            H, Ftx, Wrx, cfg.ckm.topKBeams, cfg.ckm.beamSoftmaxTemperature);

        valid = paths.mask & paths.powerLin > 0;
        power = double(paths.powerLin(valid));
        delay = double(paths.delaySec(valid));
        if isempty(power)
            power = 1;
            delay = 0;
        end
        powerWeight = power / max(sum(power), realmin);
        muDelay = sum(powerWeight .* delay);
        sigmaDelay = sqrt(sum(powerWeight .* (delay - muDelay).^2));

        losValue(r) = any(paths.isLos & paths.mask);
        pathlossDb(r) = 10*log10(norm(H, 'fro')^2 + realmin);
        meanDelay(r) = muDelay;
        rmsDelay(r) = sigmaDelay;
        txAccum = txAccum + double(beam.txProbability);
        rxAccum = rxAccum + double(beam.rxProbability);

        validIndex = find(valid);
        for l = 1:numel(validIndex)
            idx = validIndex(l);
            aodBin = localAngleBin(paths.aodAzRad(idx), nAod);
            aoaBin = localAngleBin(paths.aoaAzRad(idx), nAoa);
            delayBin = localDelayBin(paths.delaySec(idx), ...
                nDelay, cfg.ckm.maxDelaySec);
            linearBin = aodBin + (aoaBin-1)*nAod ...
                + (delayBin-1)*nAod*nAoa;
            adpAccum(linearBin) = adpAccum(linearBin) ...
                + paths.powerLin(idx);
        end
    end

    scalar(:, g) = single([ ...
        mean(losValue); mean(pathlossDb); std(pathlossDb, 0); ...
        mean(meanDelay); mean(rmsDelay); R]);
    adp(:, g) = single(adpAccum / max(sum(adpAccum), realmin));
    txBeamProbability(:, g) = single(txAccum / max(sum(txAccum), realmin));
    rxBeamProbability(:, g) = single(rxAccum / max(sum(rxAccum), realmin));
end

ckm = struct();
ckm.version = '1.0';
ckm.sceneId = scene.sceneId;
ckm.gridX = single(xGrid);
ckm.gridY = single(yGrid);
ckm.gridXY = single(gridXY);
ckm.numX = numel(xGrid);
ckm.numY = numel(yGrid);
ckm.gridResolutionM = cfg.ckm.gridResolutionM;
ckm.scalarFeature = scalar;
ckm.scalarFeatureNames = { ...
    'los_probability', 'pathloss_mean_db', 'pathloss_std_db', ...
    'mean_delay_sec', 'rms_delay_spread_sec', 'history_count'};
ckm.angularDelayPower = adp;
ckm.adpShape = uint16([nAod nAoa nDelay]);
ckm.aodEdgesRad = single(linspace(-pi, pi, nAod+1));
ckm.aoaEdgesRad = single(linspace(-pi, pi, nAoa+1));
ckm.delayEdgesSec = single(linspace(0, cfg.ckm.maxDelaySec, nDelay+1));
ckm.txBeamProbability = txBeamProbability;
ckm.rxBeamProbability = rxBeamProbability;
ckm.historySamplesPerCell = uint16(R);
ckm.constructionSeed = uint32(cfg.randomSeed + 300000*double(scene.sceneId));

end

function yaw = localHistoricalYaw(cfg)
switch lower(cfg.ueOrientationMode)
    case 'fixed'
        yaw = deg2rad(cfg.fixedUeOrientationDeg);
    case {'random', 'heading'}
        yaw = -pi + 2*pi*rand();
    otherwise
        error('CKMSRF:Orientation', 'Unsupported orientation mode.');
end
end

function bin = localAngleBin(angle, numBins)
u = (ckmsrf_wrap_to_pi(angle) + pi) / (2*pi);
bin = min(max(floor(u * numBins) + 1, 1), numBins);
end

function bin = localDelayBin(delay, numBins, maxDelay)
u = min(max(delay / max(maxDelay, realmin), 0), 1-eps);
bin = min(max(floor(u * numBins) + 1, 1), numBins);
end
