function cfg = ckmsrf_default_config()
%CKMSRF_DEFAULT_CONFIG Default configuration for the unified CKM-SRF generator.
%
% The defaults are intentionally small enough for a laptop smoke test. For
% formal experiments, start from configs/research_config.m.

cfg = struct();

% Reproducibility and output.
cfg.randomSeed = 20260713;
cfg.numScenes = 1;
cfg.outputDir = fullfile(fileparts(mfilename('fullpath')), 'output_demo');
cfg.overwrite = false;
cfg.verbose = true;

% Antenna and waveform convention:
% H(rx, tx, subcarrier, snapshot, sample).
cfg.numTxAntennas = 64;
cfg.numRxAntennas = 16;
cfg.txArraySize = [8 8];
cfg.rxArraySize = [4 4];
cfg.carrierFrequencyHz = 28e9;
cfg.subcarrierSpacingHz = 15e3;
cfg.numSubcarriers = 8;
cfg.numSnapshots = 4;
cfg.snapshotPeriodSec = 1e-3;
cfg.lightSpeed = 299792458;

% Site-specific scene.
cfg.areaX = [20 220];
cfg.areaY = [-100 100];
cfg.bsPositionM = [0 0 25];
cfg.ueHeightM = 1.5;
cfg.numScatterers = 12;
cfg.numBlockers = 3;
cfg.scattererHeightRangeM = [4 35];
cfg.scattererMagnitudeRange = [0.15 0.75];
cfg.blockerLengthRangeM = [25 70];
cfg.blockedPathAmplitudeFactor = 0.20;
cfg.referenceDistanceM = 1;
cfg.losPathLossExponent = 2.0;
cfg.nlosPathLossExponent = 2.7;
cfg.numPaths = 6;

% UE mobility and array orientation.
cfg.speedRangeMps = [20 60];
cfg.headingRangeRad = [-pi pi];
cfg.ueOrientationMode = 'heading';  % 'heading', 'fixed', or 'random'.
cfg.fixedUeOrientationDeg = 0;

% Spatially disjoint online splits.
cfg.split.blockSizeM = 40;
cfg.split.trainFraction = 0.60;
cfg.split.valFraction = 0.20;
cfg.split.testFraction = 0.20;
cfg.split.maxTrajectorySamplingTries = 3000;

% Offline historical samples used to construct the CKM. These samples are
% independent of the online train/validation/test channel realizations.
cfg.ckm.gridResolutionM = 20;
cfg.ckm.historySamplesPerCell = 4;
cfg.ckm.numAodBins = 8;
cfg.ckm.numAoaBins = 8;
cfg.ckm.numDelayBins = 6;
cfg.ckm.maxDelaySec = 2e-6;
cfg.ckm.beamSoftmaxTemperature = 0.15;
cfg.ckm.topKBeams = 4;

% Location information available at online inference.
cfg.localization.errorStdM = 0;

% Finite-pilot observation Y = A(H) + N. A semi-unitary DFT pilot matrix
% P in C^(Nt x Np) is selected from a reproducible pilot bank.
cfg.pilot.density = 0.30;
cfg.pilot.numPilots = [];  % Empty means round(density * Nt).
cfg.pilot.numPilotMatrices = 4;
cfg.pilot.snrDbChoices = [-5 0 5 10 20];

% Optional high-mobility CFO/ICI operator. Keep disabled for the original
% SRF observation model; enable only when using the generalized adjoint PBG.
cfg.cfo.enable = false;
cfg.cfo.rangeHz = [-1500 1500];

% Dataset sizes are per scene. Shards keep memory bounded.
cfg.dataset.samplesPerScene.train = 64;
cfg.dataset.samplesPerScene.val = 16;
cfg.dataset.samplesPerScene.test = 32;
cfg.dataset.shardSize = 16;

% Storage. Set storeObservation=false to synthesize pilots/noise in the
% Python data loader and substantially reduce disk use.
cfg.output.storeObservation = true;
cfg.output.storeTrueChannel = true;
cfg.output.matVersion = '-v7.3';

end
