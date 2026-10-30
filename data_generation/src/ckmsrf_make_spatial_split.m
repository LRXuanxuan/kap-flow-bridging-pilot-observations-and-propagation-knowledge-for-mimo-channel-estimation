function plan = ckmsrf_make_spatial_split(cfg, sceneId)
%CKMSRF_MAKE_SPATIAL_SPLIT Assign contiguous spatial blocks to splits.

blockSize = cfg.split.blockSizeM;
numBlockX = ceil(diff(cfg.areaX) / blockSize);
numBlockY = ceil(diff(cfg.areaY) / blockSize);
numBlocks = numBlockX * numBlockY;
if numBlocks < 3
    error('CKMSRF:Split', 'At least three spatial blocks are required.');
end

rng(cfg.randomSeed + 200000 * sceneId, 'twister');
order = randperm(numBlocks);
nTrain = max(1, floor(cfg.split.trainFraction * numBlocks));
nVal = max(1, floor(cfg.split.valFraction * numBlocks));
if nTrain + nVal >= numBlocks
    nVal = 1;
    nTrain = numBlocks - 2;
end

assignment = zeros(1, numBlocks, 'uint8');
assignment(order(1:nTrain)) = uint8(1);
assignment(order(nTrain+1:nTrain+nVal)) = uint8(2);
assignment(order(nTrain+nVal+1:end)) = uint8(3);

plan = struct();
plan.blockSizeM = blockSize;
plan.areaX = cfg.areaX;
plan.areaY = cfg.areaY;
plan.numBlockX = numBlockX;
plan.numBlockY = numBlockY;
plan.numBlocks = numBlocks;
plan.assignment = assignment;
plan.splitNames = {'train', 'val', 'test'};

end
