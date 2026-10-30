function trajectory = ckmsrf_sample_trajectory(cfg, splitPlan, splitName, sampleSeed)
%CKMSRF_SAMPLE_TRAJECTORY Sample a trajectory wholly contained in one split.

splitCode = localSplitCode(splitName);
rng(sampleSeed, 'twister');
t = cfg.timeAxisSec(:);

for attempt = 1:cfg.split.maxTrajectorySamplingTries
    x0 = cfg.areaX(1) + diff(cfg.areaX) * rand();
    y0 = cfg.areaY(1) + diff(cfg.areaY) * rand();
    speed = cfg.speedRangeMps(1) + diff(cfg.speedRangeMps) * rand();
    heading = cfg.headingRangeRad(1) + diff(cfg.headingRangeRad) * rand();
    velocity = [speed*cos(heading), speed*sin(heading), 0];
    positionXY = [x0 + velocity(1)*t, y0 + velocity(2)*t];

    inside = all(positionXY(:,1) >= cfg.areaX(1)) ...
        && all(positionXY(:,1) <= cfg.areaX(2)) ...
        && all(positionXY(:,2) >= cfg.areaY(1)) ...
        && all(positionXY(:,2) <= cfg.areaY(2));
    if ~inside
        continue;
    end

    blockId = ckmsrf_position_block_id(positionXY, splitPlan);
    if ~all(splitPlan.assignment(blockId) == splitCode)
        continue;
    end

    switch lower(cfg.ueOrientationMode)
        case 'heading'
            yawDeg = repmat(rad2deg(heading), numel(t), 1);
        case 'fixed'
            yawDeg = repmat(cfg.fixedUeOrientationDeg, numel(t), 1);
        case 'random'
            yawDeg = repmat(-180 + 360*rand(), numel(t), 1);
        otherwise
            error('CKMSRF:Orientation', 'Unsupported orientation mode.');
    end

    trajectory = struct();
    trajectory.timeSec = t;
    trajectory.positionXYZ = [positionXY repmat(cfg.ueHeightM, numel(t), 1)];
    trajectory.velocityXYZ = repmat(velocity, numel(t), 1);
    trajectory.speedMps = speed;
    trajectory.headingRad = heading;
    trajectory.ueOrientationDeg = yawDeg;
    trajectory.blockId = uint16(blockId(:));
    return;
end

error('CKMSRF:Trajectory', ...
    'Could not sample a trajectory inside the requested spatial split.');

end

function code = localSplitCode(name)
switch lower(name)
    case 'train'
        code = uint8(1);
    case {'val', 'validation'}
        code = uint8(2);
    case 'test'
        code = uint8(3);
    otherwise
        error('CKMSRF:Split', 'Unknown split name: %s', name);
end
end
