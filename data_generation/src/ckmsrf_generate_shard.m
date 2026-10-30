function shard = ckmsrf_generate_shard( ...
    scene, ckm, splitPlan, splitName, numSamples, cfg, pilotBank, seedBase)
%CKMSRF_GENERATE_SHARD Generate one bounded-memory online data shard.

Nr = cfg.numRxAntennas;
Nt = cfg.numTxAntennas;
Nsc = cfg.numSubcarriers;
T = cfg.numSnapshots;
L = cfg.numPaths;
Np = cfg.pilot.numPilots;
Fscalar = size(ckm.scalarFeature, 1);
Fadp = size(ckm.angularDelayPower, 1);
K = cfg.ckm.topKBeams;
N = numSamples;

shard = struct();
shard.metadata.version = '1.0';
shard.metadata.splitName = splitName;
shard.metadata.sceneId = scene.sceneId;
shard.metadata.dimensionOrderH = 'rx,tx,subcarrier,snapshot,sample';
shard.metadata.dimensionOrderY = 'rx,pilot,subcarrier,snapshot,sample';

if cfg.output.storeTrueChannel
    shard.H_true = complex(zeros(Nr,Nt,Nsc,T,N,'single'), ...
        zeros(Nr,Nt,Nsc,T,N,'single'));
else
    shard.H_true = [];
end
if cfg.output.storeObservation
    shard.Y_pilot = complex(zeros(Nr,Np,Nsc,T,N,'single'), ...
        zeros(Nr,Np,Nsc,T,N,'single'));
else
    shard.Y_pilot = [];
end
shard.channelScaleLin = zeros(Nsc,T,N,'single');

shard.state.positionXYZ = zeros(3,T,N,'single');
shard.state.velocityXYZ = zeros(3,T,N,'single');
shard.state.ueOrientationDeg = zeros(1,T,N,'single');
shard.state.timeSec = single(cfg.timeAxisSec(:));
shard.state.blockId = zeros(T,N,'uint16');

shard.context.observedPositionXYZ = zeros(3,T,N,'single');
shard.context.gridIndices = zeros(4,T,N,'uint32');
shard.context.gridWeights = zeros(4,T,N,'single');
shard.context.scalarFeature = zeros(Fscalar,T,N,'single');
shard.context.angularDelayPower = zeros(Fadp,T,N,'single');
shard.context.txBeamProbability = zeros(Nt,T,N,'single');
shard.context.rxBeamProbability = zeros(Nr,T,N,'single');

shard.truth.pathId = zeros(L,T,N,'int32');
shard.truth.pathMask = false(L,T,N);
shard.truth.pathGain = complex(zeros(L,T,N,'single'), zeros(L,T,N,'single'));
shard.truth.pathPowerLin = zeros(L,T,N,'single');
shard.truth.delaySec = zeros(L,T,N,'single');
shard.truth.aodAzRad = zeros(L,T,N,'single');
shard.truth.aodElRad = zeros(L,T,N,'single');
shard.truth.aoaAzRad = zeros(L,T,N,'single');
shard.truth.aoaElRad = zeros(L,T,N,'single');
shard.truth.dopplerHz = zeros(L,T,N,'single');

shard.aux.txBeamTopK = zeros(K,T,N,'uint16');
shard.aux.rxBeamTopK = zeros(K,T,N,'uint16');
shard.aux.txBeamTopKProbability = zeros(K,T,N,'single');
shard.aux.rxBeamTopKProbability = zeros(K,T,N,'single');
shard.aux.bestBeamGain = zeros(T,N,'single');

shard.observation.pilotId = zeros(1,N,'uint16');
shard.observation.snrDb = zeros(1,N,'single');
shard.observation.cfoHz = zeros(1,N,'single');
shard.observation.noiseVar = zeros(1,N,'single');
shard.sampleSeed = zeros(1,N,'uint32');

centerSubcarrier = floor((Nsc+1)/2);
Ftx = ckmsrf_dft_upa_codebook(cfg.txArraySize);
Wrx = ckmsrf_dft_upa_codebook(cfg.rxArraySize);

for n = 1:N
    sampleSeed = seedBase + n;
    trajectory = ckmsrf_sample_trajectory( ...
        cfg, splitPlan, splitName, sampleSeed);
    sequence = ckmsrf_generate_channel_sequence( ...
        scene, trajectory, cfg, sampleSeed + 10000000);

    rng(sampleSeed + 20000000, 'twister');
    observedPosition = trajectory.positionXYZ;
    observedPosition(:,1:2) = observedPosition(:,1:2) ...
        + cfg.localization.errorStdM * randn(T,2);
    observedPosition(:,1) = min(max(observedPosition(:,1),cfg.areaX(1)),cfg.areaX(2));
    observedPosition(:,2) = min(max(observedPosition(:,2),cfg.areaY(1)),cfg.areaY(2));

    for t = 1:T
        query = ckmsrf_query_ckm(ckm, observedPosition(t,1:2));
        shard.context.observedPositionXYZ(:,t,n) = single(observedPosition(t,:));
        shard.context.gridIndices(:,t,n) = query.gridIndices;
        shard.context.gridWeights(:,t,n) = query.weights;
        shard.context.scalarFeature(:,t,n) = query.scalarFeature;
        shard.context.angularDelayPower(:,t,n) = query.angularDelayPower;
        shard.context.txBeamProbability(:,t,n) = query.txBeamProbability;
        shard.context.rxBeamProbability(:,t,n) = query.rxBeamProbability;

        Hcenter = double(sequence.H(:,:,centerSubcarrier,t));
        beam = ckmsrf_beam_features( ...
            Hcenter,Ftx,Wrx,K,cfg.ckm.beamSoftmaxTemperature);
        shard.aux.txBeamTopK(:,t,n) = localPadIndex(beam.txTopKIndex,K);
        shard.aux.rxBeamTopK(:,t,n) = localPadIndex(beam.rxTopKIndex,K);
        shard.aux.txBeamTopKProbability(:,t,n) = ...
            localPadSingle(beam.txTopKProbability,K);
        shard.aux.rxBeamTopKProbability(:,t,n) = ...
            localPadSingle(beam.rxTopKProbability,K);
        shard.aux.bestBeamGain(t,n) = beam.bestBeamGain;
    end

    rng(sampleSeed + 30000000, 'twister');
    pilotId = randi(cfg.pilot.numPilotMatrices);
    snrChoice = randi(numel(cfg.pilot.snrDbChoices));
    snrDb = cfg.pilot.snrDbChoices(snrChoice);
    if cfg.cfo.enable
        cfoHz = cfg.cfo.rangeHz(1) + diff(cfg.cfo.rangeHz)*rand();
    else
        cfoHz = 0;
    end
    P = double(pilotBank.P(:,:,pilotId));
    Y0 = ckmsrf_apply_operator(sequence.H,P,cfoHz,cfg);
    [Y,noiseVar] = ckmsrf_add_awgn(Y0,snrDb,sampleSeed+40000000);

    if cfg.output.storeTrueChannel
        shard.H_true(:,:,:,:,n) = sequence.H;
    end
    if cfg.output.storeObservation
        shard.Y_pilot(:,:,:,:,n) = single(Y);
    end
    shard.channelScaleLin(:,:,n) = sequence.scaleLin;
    shard.state.positionXYZ(:,:,n) = single(trajectory.positionXYZ.');
    shard.state.velocityXYZ(:,:,n) = single(trajectory.velocityXYZ.');
    shard.state.ueOrientationDeg(:,:,n) = single(trajectory.ueOrientationDeg.');
    shard.state.blockId(:,n) = trajectory.blockId;

    fields = fieldnames(sequence.truth);
    for f = 1:numel(fields)
        shard.truth.(fields{f})(:,:,n) = sequence.truth.(fields{f});
    end

    shard.observation.pilotId(n) = uint16(pilotId);
    shard.observation.snrDb(n) = single(snrDb);
    shard.observation.cfoHz(n) = single(cfoHz);
    shard.observation.noiseVar(n) = single(noiseVar);
    shard.sampleSeed(n) = uint32(sampleSeed);
end

end

function out = localPadIndex(value,K)
out = zeros(K,1,'uint16');
n = min(K,numel(value));
out(1:n) = uint16(value(1:n));
end

function out = localPadSingle(value,K)
out = zeros(K,1,'single');
n = min(K,numel(value));
out(1:n) = single(value(1:n));
end
