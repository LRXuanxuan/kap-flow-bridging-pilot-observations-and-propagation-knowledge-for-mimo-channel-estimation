clear; clc;

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir,'src'));

cfg = struct();
cfg.outputDir = fullfile(rootDir,'output_demo');
cfg.overwrite = true;
cfg.numScenes = 1;
cfg.numSubcarriers = 4;
cfg.numSnapshots = 3;
cfg.ckm.historySamplesPerCell = 2;
cfg.dataset.samplesPerScene.train = 12;
cfg.dataset.samplesPerScene.val = 6;
cfg.dataset.samplesPerScene.test = 6;
cfg.dataset.shardSize = 6;
cfg.output.storeObservation = true;
cfg.cfo.enable = false;

manifest = generate_ckm_srf_dataset(cfg);
disp(manifest.totalSamples);

fprintf('\nNext, run tests/run_all_tests.m for physics and schema checks.\n');
