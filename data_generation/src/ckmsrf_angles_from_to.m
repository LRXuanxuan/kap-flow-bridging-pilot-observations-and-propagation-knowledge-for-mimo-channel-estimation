function [az, el, unitVec] = ckmsrf_angles_from_to(pFrom, pTo)
%CKMSRF_ANGLES_FROM_TO Direction from pFrom toward pTo.
v = pTo(:).' - pFrom(:).';
d = norm(v);
if d < eps
    unitVec = [1 0 0];
else
    unitVec = v / d;
end
az = atan2(unitVec(2), unitVec(1));
el = asin(max(min(unitVec(3), 1), -1));
end
