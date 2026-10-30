function paths = ckmsrf_compute_paths(scene, positionXYZ, ueYawRad, cfg)
%CKMSRF_COMPUTE_PATHS Compute LoS and single-bounce geometric paths.
%
% Direction convention:
%   AoD points from the BS toward the UE/scatterer.
%   AoA points from the UE toward the BS/scatterer (source direction in the
%   UE-local coordinate system).

ue = double(positionXYZ(:).');
bs = double(scene.bsPositionM(:).');
S = size(scene.scattererPositionM, 1);
Lall = S + 1;

paths = struct();
paths.pathId = int32((0:S).');
paths.mask = false(Lall, 1);
paths.delaySec = zeros(Lall, 1);
paths.powerLin = zeros(Lall, 1);
paths.amplitude = complex(zeros(Lall, 1));
paths.aodAzRad = zeros(Lall, 1);
paths.aodElRad = zeros(Lall, 1);
paths.aoaAzRad = zeros(Lall, 1);
paths.aoaElRad = zeros(Lall, 1);
paths.arrivalUnitTowardSource = zeros(Lall, 3);
paths.isLos = false(Lall, 1);

lambda = cfg.lightSpeed / cfg.carrierFrequencyHz;
referenceAmp = lambda / (4*pi*cfg.referenceDistanceM);

% LoS path: AoD BS->UE; AoA UE->BS.
[aodAz, aodEl] = ckmsrf_angles_from_to(bs, ue);
[aoaAzGlobal, aoaEl, arrivalUnit] = ckmsrf_angles_from_to(ue, bs);
aoaAzLocal = ckmsrf_wrap_to_pi(aoaAzGlobal - ueYawRad);
d = norm(ue - bs);
blocked = localIsBlocked(bs(1:2), ue(1:2), scene.blockers);
if ~blocked
    amp = referenceAmp * ...
        (cfg.referenceDistanceM / max(d, cfg.referenceDistanceM)) ^ ...
        (cfg.losPathLossExponent / 2);
    paths.mask(1) = true;
    paths.amplitude(1) = amp;
    paths.powerLin(1) = abs(amp)^2;
end
paths.delaySec(1) = d / cfg.lightSpeed;
paths.aodAzRad(1) = aodAz;
paths.aodElRad(1) = aodEl;
paths.aoaAzRad(1) = aoaAzLocal;
paths.aoaElRad(1) = aoaEl;
paths.arrivalUnitTowardSource(1, :) = arrivalUnit;
paths.isLos(1) = true;

% Single-bounce paths via fixed site scatterers.
for s = 1:S
    idx = s + 1;
    sc = double(scene.scattererPositionM(s, :));
    [aodAz, aodEl] = ckmsrf_angles_from_to(bs, sc);
    [aoaAzGlobal, aoaEl, arrivalUnit] = ckmsrf_angles_from_to(ue, sc);
    aoaAzLocal = ckmsrf_wrap_to_pi(aoaAzGlobal - ueYawRad);
    d1 = norm(sc - bs);
    d2 = norm(ue - sc);
    dTotal = d1 + d2;

    blockFactor = 1;
    if localIsBlocked(bs(1:2), sc(1:2), scene.blockers)
        blockFactor = blockFactor * cfg.blockedPathAmplitudeFactor;
    end
    if localIsBlocked(sc(1:2), ue(1:2), scene.blockers)
        blockFactor = blockFactor * cfg.blockedPathAmplitudeFactor;
    end

    amp = scene.scattererReflection(s) * blockFactor * referenceAmp * ...
        (cfg.referenceDistanceM / max(dTotal, cfg.referenceDistanceM)) ^ ...
        (cfg.nlosPathLossExponent / 2);

    paths.mask(idx) = abs(amp) > 0;
    paths.amplitude(idx) = amp;
    paths.powerLin(idx) = abs(amp)^2;
    paths.delaySec(idx) = dTotal / cfg.lightSpeed;
    paths.aodAzRad(idx) = aodAz;
    paths.aodElRad(idx) = aodEl;
    paths.aoaAzRad(idx) = aoaAzLocal;
    paths.aoaElRad(idx) = aoaEl;
    paths.arrivalUnitTowardSource(idx, :) = arrivalUnit;
end

end

function tf = localIsBlocked(p1, p2, blockers)
tf = false;
for b = 1:size(blockers, 3)
    q1 = blockers(1, :, b);
    q2 = blockers(2, :, b);
    if localSegmentsIntersect(p1, p2, q1, q2)
        tf = true;
        return;
    end
end
end

function tf = localSegmentsIntersect(p1, p2, q1, q2)
o1 = localOrientation(p1, p2, q1);
o2 = localOrientation(p1, p2, q2);
o3 = localOrientation(q1, q2, p1);
o4 = localOrientation(q1, q2, p2);
tol = 1e-10;
tf = (o1 * o2 < -tol) && (o3 * o4 < -tol);
end

function value = localOrientation(a, b, c)
value = (b(1)-a(1)) * (c(2)-a(2)) ...
    - (b(2)-a(2)) * (c(1)-a(1));
end
