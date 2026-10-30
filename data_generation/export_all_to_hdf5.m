function export_all_to_hdf5(inputDir, outputDir)
%EXPORT_ALL_TO_HDF5 Batch-export generated .mat files to plain HDF5.
%
% The per-file exporters export_scene_to_hdf5.m / export_shard_to_hdf5.m are
% the canonical MATLAB -> HDF5 bridge used by the KAP-Flow data loader. This
% helper only loops over a generator output directory:
%
%   export_all_to_hdf5('<datasets>/ckm_srf_64x16');
%   export_all_to_hdf5('<datasets>/ckm_srf_64x16', '<datasets>/ckm_srf_64x16_h5');
%
% Output defaults to "<inputDir>_h5". Existing target files are replaced.

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir, 'src'));

if nargin < 1 || isempty(inputDir)
    error('CKMSRF:Export', 'inputDir is required.');
end
if nargin < 2 || isempty(outputDir)
    outputDir = [inputDir '_h5'];
end
if ~exist(inputDir, 'dir')
    error('CKMSRF:Export', 'Input directory not found: %s', inputDir);
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

sceneFiles = dir(fullfile(inputDir, 'scene_*_ckm.mat'));
for i = 1:numel(sceneFiles)
    source = fullfile(inputDir, sceneFiles(i).name);
    [~, stem] = fileparts(sceneFiles(i).name);
    destination = fullfile(outputDir, [stem '.h5']);
    export_scene_to_hdf5(source, destination);
    fprintf('[scene] %s\n', destination);
end

shardFiles = dir(fullfile(inputDir, 'scene_*_shard_*.mat'));
for i = 1:numel(shardFiles)
    source = fullfile(inputDir, shardFiles(i).name);
    [~, stem] = fileparts(shardFiles(i).name);
    destination = fullfile(outputDir, [stem '.h5']);
    export_shard_to_hdf5(source, destination);
    fprintf('[shard] %s\n', destination);
end

fprintf('Exported %d scene file(s) and %d shard(s) to %s\n', ...
    numel(sceneFiles), numel(shardFiles), outputDir);

end
