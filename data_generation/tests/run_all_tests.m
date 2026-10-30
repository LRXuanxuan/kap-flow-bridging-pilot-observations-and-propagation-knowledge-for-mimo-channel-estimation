clear; clc;

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
addpath(fullfile(rootDir,'src'));

fprintf('Running CKM-SRF generator tests...\n');

cfg = ckmsrf_default_config();
cfg.numTxAntennas = 8;
cfg.numRxAntennas = 4;
cfg.txArraySize = [2 4];
cfg.rxArraySize = [2 2];
cfg.numSubcarriers = 3;
cfg.numSnapshots = 2;
cfg.numScatterers = 5;
cfg.numPaths = 4;
cfg.numBlockers = 2;
cfg.areaX = [20 100];
cfg.areaY = [-40 40];
cfg.ckm.gridResolutionM = 40;
cfg.ckm.historySamplesPerCell = 2;
cfg.ckm.numAodBins = 4;
cfg.ckm.numAoaBins = 4;
cfg.ckm.numDelayBins = 3;
cfg.split.blockSizeM = 20;
cfg.pilot.numPilots = 3;
cfg.pilot.numPilotMatrices = 2;
cfg.speedRangeMps = [5 10];
cfg.cfo.enable = true;
cfg = ckmsrf_validate_config(cfg);

scene = ckmsrf_create_scene(cfg,1);
splitPlan = ckmsrf_make_spatial_split(cfg,1);
pilotBank = ckmsrf_generate_pilot_bank(cfg,1);
Ftx = ckmsrf_dft_upa_codebook(cfg.txArraySize);
Wrx = ckmsrf_dft_upa_codebook(cfg.rxArraySize);
ckm = ckmsrf_build_ckm(scene,cfg,Ftx,Wrx);

%% Test 1: spatial split assignments are disjoint and complete.
assert(all(ismember(splitPlan.assignment,[1 2 3])));
assert(~isempty(find(splitPlan.assignment==1,1)));
assert(~isempty(find(splitPlan.assignment==2,1)));
assert(~isempty(find(splitPlan.assignment==3,1)));
fprintf('  [pass] spatial split assignment\n');

%% Test 2: CKM interpolation is normalized and finite.
query = ckmsrf_query_ckm(ckm,mean([cfg.areaX;cfg.areaY],2).');
assert(abs(sum(double(query.weights))-1) < 1e-6);
assert(abs(sum(double(query.angularDelayPower))-1) < 1e-5);
assert(abs(sum(double(query.txBeamProbability))-1) < 1e-5);
assert(abs(sum(double(query.rxBeamProbability))-1) < 1e-5);
assert(all(isfinite(double(query.scalarFeature))));
fprintf('  [pass] continuous CKM query\n');

%% Test 3: trajectories remain wholly inside the requested split.
trajectory = ckmsrf_sample_trajectory(cfg,splitPlan,'train',12345);
ids = ckmsrf_position_block_id(trajectory.positionXYZ(:,1:2),splitPlan);
assert(all(splitPlan.assignment(ids)==1));
fprintf('  [pass] trajectory split containment\n');

%% Test 4: channel dimensions and fixed path identity.
sequence1 = ckmsrf_generate_channel_sequence(scene,trajectory,cfg,23456);
sequence2 = ckmsrf_generate_channel_sequence(scene,trajectory,cfg,23456);
assert(isequal(size(sequence1.H), ...
    [cfg.numRxAntennas cfg.numTxAntennas ...
     cfg.numSubcarriers cfg.numSnapshots]));
assert(isequal(sequence1.truth.pathId(:,1),sequence1.truth.pathId(:,end)));
assert(isequal(sequence1.H,sequence2.H));
assert(all(isfinite(double(sequence1.H(:)))));
fprintf('  [pass] coherent reproducible channel sequence\n');

%% Test 5: pilots are semi-unitary.
P = double(pilotBank.P(:,:,1));
assert(norm(P'*P-eye(cfg.pilot.numPilots),'fro') < 1e-6);
fprintf('  [pass] semi-unitary finite pilot matrix\n');

%% Test 6: operator output has Nr x Np x Nsc x T dimensions.
cfoHz = 700;
Y = ckmsrf_apply_operator(sequence1.H,P,cfoHz,cfg);
assert(isequal(size(Y), ...
    [cfg.numRxAntennas cfg.pilot.numPilots ...
     cfg.numSubcarriers cfg.numSnapshots]));
fprintf('  [pass] finite-pilot/CFO observation dimensions\n');

%% Test 7: numerical adjoint identity <A(H),Z> = <H,A*(Z)>.
rng(34567,'twister');
Z = randn(size(Y)) + 1j*randn(size(Y));
AH = ckmsrf_apply_operator(sequence1.H,P,cfoHz,cfg);
AstarZ = ckmsrf_apply_adjoint(Z,P,cfoHz,cfg);
lhs = sum(conj(AH(:)).*Z(:));
rhs = sum(conj(double(sequence1.H(:))).*AstarZ(:));
relativeError = abs(lhs-rhs)/max([abs(lhs),abs(rhs),1]);
assert(relativeError < 1e-10);
fprintf('  [pass] generalized operator adjoint (error %.3e)\n',relativeError);

%% Test 8: no-CFO operator reduces exactly to Y=H*P.
cfgNoCfo = cfg;
cfgNoCfo.cfo.enable = false;
Ysimple = ckmsrf_apply_operator(sequence1.H,P,0,cfgNoCfo);
Ymanual = zeros(size(Ysimple));
for t = 1:cfg.numSnapshots
    for k = 1:cfg.numSubcarriers
        Ymanual(:,:,k,t) = double(sequence1.H(:,:,k,t))*P;
    end
end
assert(norm(Ysimple(:)-Ymanual(:))/max(norm(Ymanual(:)),realmin) < 1e-12);
fprintf('  [pass] original SRF observation reduction\n');

%% Test 9: one complete shard respects the schema.
shard = ckmsrf_generate_shard( ...
    scene,ckm,splitPlan,'train',2,cfg,pilotBank,45678);
assert(isequal(size(shard.H_true), ...
    [cfg.numRxAntennas cfg.numTxAntennas cfg.numSubcarriers ...
     cfg.numSnapshots 2]));
assert(isequal(size(shard.Y_pilot), ...
    [cfg.numRxAntennas cfg.pilot.numPilots cfg.numSubcarriers ...
     cfg.numSnapshots 2]));
assert(all(shard.metadata.sceneId==scene.sceneId));
fprintf('  [pass] shard schema\n');

fprintf('\nAll CKM-SRF tests passed.\n');
