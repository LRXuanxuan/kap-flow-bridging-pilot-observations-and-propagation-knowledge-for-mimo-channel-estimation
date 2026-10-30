function query = ckmsrf_query_ckm(ckm, positionXY)
%CKMSRF_QUERY_CKM Bilinearly interpolate continuous CKM features.

x = double(positionXY(1));
y = double(positionXY(2));
xGrid = double(ckm.gridX(:).');
yGrid = double(ckm.gridY(:).');
x = min(max(x, xGrid(1)), xGrid(end));
y = min(max(y, yGrid(1)), yGrid(end));

[ix0, ix1, wx] = localAxisBracket(xGrid, x);
[iy0, iy1, wy] = localAxisBracket(yGrid, y);

indices = [ ...
    localIndex(ix0, iy0, ckm.numY); ...
    localIndex(ix1, iy0, ckm.numY); ...
    localIndex(ix0, iy1, ckm.numY); ...
    localIndex(ix1, iy1, ckm.numY)];
weights = [(1-wx)*(1-wy); wx*(1-wy); (1-wx)*wy; wx*wy];

scalar = double(ckm.scalarFeature(:, indices)) * weights;
adp = double(ckm.angularDelayPower(:, indices)) * weights;
txProb = double(ckm.txBeamProbability(:, indices)) * weights;
rxProb = double(ckm.rxBeamProbability(:, indices)) * weights;
adp = adp / max(sum(adp), realmin);
txProb = txProb / max(sum(txProb), realmin);
rxProb = rxProb / max(sum(rxProb), realmin);

query = struct();
query.gridIndices = uint32(indices);
query.weights = single(weights);
query.scalarFeature = single(scalar);
query.angularDelayPower = single(adp);
query.txBeamProbability = single(txProb);
query.rxBeamProbability = single(rxProb);

end

function [i0, i1, w] = localAxisBracket(grid, value)
if value <= grid(1)
    i0 = 1; i1 = 1; w = 0;
    return;
end
if value >= grid(end)
    i0 = numel(grid); i1 = i0; w = 0;
    return;
end
i1 = find(grid >= value, 1, 'first');
i0 = i1 - 1;
w = (value - grid(i0)) / max(grid(i1) - grid(i0), realmin);
end

function idx = localIndex(ix, iy, numY)
idx = iy + (ix-1) * numY;
end
