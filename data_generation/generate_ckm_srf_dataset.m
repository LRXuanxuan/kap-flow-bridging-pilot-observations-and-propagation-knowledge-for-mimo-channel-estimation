function manifest = generate_ckm_srf_dataset(userCfg)
%GENERATE_CKM_SRF_DATASET Unified CKM-conditioned SRF dataset generator.
%
% This function creates independent site scenes, constructs each CKM from
% historical samples, generates spatially disjoint online splits, and forms
% finite-pilot observations with optional CFO/ICI.

rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir, 'src'));

cfg0 = ckmsrf_default_config();
if nargin < 1
    userCfg = struct();
end
cfg = ckmsrf_merge_struct(cfg0, userCfg);
cfg = ckmsrf_validate_config(cfg);

if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
elseif ~cfg.overwrite
    existing = dir(fullfile(cfg.outputDir, '*.mat'));
    if ~isempty(existing)
        error('CKMSRF:OutputExists', ...
            ['Output directory contains MAT files. Set cfg.overwrite=true ', ...
             'or choose another outputDir.']);
    end
end

if cfg.verbose
    fprintf('CKM-SRF unified dataset generation\n');
    fprintf('  output        : %s\n', cfg.outputDir);
    fprintf('  H dimensions  : %d x %d x %d x %d\n', ...
        cfg.numRxAntennas,cfg.numTxAntennas, ...
        cfg.numSubcarriers,cfg.numSnapshots);
    fprintf('  pilot density : %.4f (%d/%d)\n', ...
        cfg.pilot.density,cfg.pilot.numPilots,cfg.numTxAntennas);
    fprintf('  CFO/ICI       : %d\n', cfg.cfo.enable);
end

manifest = struct();
manifest.version = '1.0';
manifest.createdBy = 'generate_ckm_srf_dataset';
manifest.dimensionOrderH = 'rx,tx,subcarrier,snapshot,sample';
manifest.dimensionOrderY = 'rx,pilot,subcarrier,snapshot,sample';
manifest.config = cfg;
manifest.files = {};
manifest.sceneFiles = {};
manifest.totalSamples = struct('train',0,'val',0,'test',0);

splitNames = {'train','val','test'};
Ftx = ckmsrf_dft_upa_codebook(cfg.txArraySize);
Wrx = ckmsrf_dft_upa_codebook(cfg.rxArraySize);

for sceneId = 1:cfg.numScenes
    if cfg.verbose
        fprintf('\nScene %d/%d: creating geometry and historical CKM...\n', ...
            sceneId,cfg.numScenes);
    end
    scene = ckmsrf_create_scene(cfg,sceneId);
    splitPlan = ckmsrf_make_spatial_split(cfg,sceneId);
    pilotBank = ckmsrf_generate_pilot_bank(cfg,sceneId);
    ckm = ckmsrf_build_ckm(scene,cfg,Ftx,Wrx);

    scenePackage = struct();
    scenePackage.scene = scene;
    scenePackage.ckm = ckm;
    scenePackage.splitPlan = splitPlan;
    scenePackage.pilotBank = pilotBank;
    scenePackage.Ftx = single(Ftx);
    scenePackage.Wrx = single(Wrx);
    sceneFileName = sprintf('scene_%02d_ckm.mat',sceneId);
    sceneFile = fullfile(cfg.outputDir,sceneFileName);
    save(sceneFile,'scenePackage',cfg.output.matVersion);
    manifest.sceneFiles{end+1} = sceneFileName;

    for s = 1:numel(splitNames)
        splitName = splitNames{s};
        total = cfg.dataset.samplesPerScene.(splitName);
        numShards = ceil(total/cfg.dataset.shardSize);
        generated = 0;

        for shardId = 1:numShards
            shardCount = min(cfg.dataset.shardSize,total-generated);
            seedBase = cfg.randomSeed + 100000000*sceneId ...
                + 1000000*s + 10000*shardId;
            if cfg.verbose
                fprintf('  %s shard %d/%d: %d samples\n', ...
                    splitName,shardId,numShards,shardCount);
            end
            shard = ckmsrf_generate_shard( ...
                scene,ckm,splitPlan,splitName,shardCount, ...
                cfg,pilotBank,seedBase);
            shardFileName = sprintf( ...
                'scene_%02d_%s_shard_%03d.mat',sceneId,splitName,shardId);
            shardFile = fullfile(cfg.outputDir,shardFileName);
            save(shardFile,'shard',cfg.output.matVersion);
            manifest.files{end+1} = shardFileName;
            generated = generated + shardCount;
        end
        manifest.totalSamples.(splitName) = ...
            manifest.totalSamples.(splitName) + generated;
    end
end

manifestFile = fullfile(cfg.outputDir,'manifest.mat');
save(manifestFile,'manifest',cfg.output.matVersion);

if cfg.verbose
    fprintf('\nGeneration completed.\n');
    fprintf('  train/val/test = %d/%d/%d\n', ...
        manifest.totalSamples.train,manifest.totalSamples.val, ...
        manifest.totalSamples.test);
    fprintf('  manifest       = %s\n',manifestFile);
end

end
