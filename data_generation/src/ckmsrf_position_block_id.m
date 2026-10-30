function blockId = ckmsrf_position_block_id(positionXY, plan)
%CKMSRF_POSITION_BLOCK_ID Spatial block identifier for one or more points.

x = positionXY(:, 1);
y = positionXY(:, 2);
ix = floor((x - plan.areaX(1)) / plan.blockSizeM) + 1;
iy = floor((y - plan.areaY(1)) / plan.blockSizeM) + 1;
ix = min(max(ix, 1), plan.numBlockX);
iy = min(max(iy, 1), plan.numBlockY);
blockId = iy + (ix - 1) * plan.numBlockY;

end
