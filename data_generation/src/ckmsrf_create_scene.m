function scene = ckmsrf_create_scene(cfg, sceneId)
%CKMSRF_CREATE_SCENE Create a reproducible site-specific geometry.

seed = cfg.randomSeed + 100000 * sceneId;
rng(seed, 'twister');

scene = struct();
scene.sceneId = uint16(sceneId);
scene.seed = uint32(seed);
scene.bsPositionM = double(cfg.bsPositionM(:).');
scene.areaX = double(cfg.areaX);
scene.areaY = double(cfg.areaY);

S = cfg.numScatterers;
scattererXY = zeros(S, 2);
scattererXY(:, 1) = cfg.areaX(1) + diff(cfg.areaX) * rand(S, 1);
scattererXY(:, 2) = cfg.areaY(1) + diff(cfg.areaY) * rand(S, 1);
scattererZ = cfg.scattererHeightRangeM(1) + ...
    diff(cfg.scattererHeightRangeM) * rand(S, 1);
reflectionMag = cfg.scattererMagnitudeRange(1) + ...
    diff(cfg.scattererMagnitudeRange) * rand(S, 1);
reflectionPhase = -pi + 2*pi*rand(S, 1);

scene.scattererPositionM = [scattererXY scattererZ];
scene.scattererReflection = reflectionMag .* exp(1j * reflectionPhase);

% Blockers are 2-D line segments. They create site-specific LoS/NLoS
% boundaries while remaining toolbox independent.
B = cfg.numBlockers;
scene.blockers = zeros(2, 2, B);
for b = 1:B
    center = [ ...
        cfg.areaX(1) + diff(cfg.areaX) * rand(), ...
        cfg.areaY(1) + diff(cfg.areaY) * rand()];
    len = cfg.blockerLengthRangeM(1) + diff(cfg.blockerLengthRangeM) * rand();
    angle = -pi + 2*pi*rand();
    delta = 0.5 * len * [cos(angle) sin(angle)];
    scene.blockers(1, :, b) = center - delta;
    scene.blockers(2, :, b) = center + delta;
end

end
