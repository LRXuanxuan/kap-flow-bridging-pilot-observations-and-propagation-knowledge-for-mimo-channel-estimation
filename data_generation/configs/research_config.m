function cfg = research_config(outputDir)
%RESEARCH_CONFIG Suggested starting point for formal CKM-SRF experiments.
%
% Adjust sample counts after checking disk/memory usage. With 48
% subcarriers and 8 snapshots, storing H_true for 10,000 trajectories is
% approximately 29.3 GiB in complex single precision before HDF5 overhead.

cfg = struct();
cfg.outputDir = outputDir;
cfg.overwrite = false;
cfg.numScenes = 3;
cfg.numSubcarriers = 48;
cfg.numSnapshots = 8;
cfg.snapshotPeriodSec = 1e-3;

cfg.ckm.gridResolutionM = 10;
cfg.ckm.historySamplesPerCell = 32;
cfg.ckm.numAodBins = 16;
cfg.ckm.numAoaBins = 16;
cfg.ckm.numDelayBins = 12;
cfg.ckm.topKBeams = 8;

cfg.pilot.density = 0.30;
cfg.pilot.numPilots = [];
cfg.pilot.numPilotMatrices = 16;
cfg.pilot.snrDbChoices = -10:5:30;

cfg.dataset.samplesPerScene.train = 10000;
cfg.dataset.samplesPerScene.val = 1000;
cfg.dataset.samplesPerScene.test = 2000;
cfg.dataset.shardSize = 250;

% Recommended: synthesize Y in the training loader to save disk and sample
% new pilot/SNR/noise conditions every epoch.
cfg.output.storeObservation = false;
cfg.output.storeTrueChannel = true;

% Keep false for the original SRF paper. Enable only with generalized PBG.
cfg.cfo.enable = false;
cfg.localization.errorStdM = 0;

end
