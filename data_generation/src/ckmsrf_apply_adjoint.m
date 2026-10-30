function Hgrad = ckmsrf_apply_adjoint(Z, P, cfoHz, cfg)
%CKMSRF_APPLY_ADJOINT Adjoint of ckmsrf_apply_operator.
%
% Z: Nr x Np x Nsc x T
% Hgrad: Nr x Nt x Nsc x T

Nr = cfg.numRxAntennas;
Nt = cfg.numTxAntennas;
Np = size(P, 2);
Nsc = cfg.numSubcarriers;
T = cfg.numSnapshots;
if size(Z,1) ~= Nr || size(Z,2) ~= Np ...
        || size(Z,3) ~= Nsc || size(Z,4) ~= T
    error('CKMSRF:Adjoint', 'Z has incompatible dimensions.');
end

if cfg.cfo.enable
    C = ckmsrf_cfo_matrix(Nsc, cfoHz / cfg.subcarrierSpacingHz);
else
    C = eye(Nsc);
end

Hgrad = complex(zeros(Nr, Nt, Nsc, T), zeros(Nr, Nt, Nsc, T));
for t = 1:T
    phase = exp(1j * 2*pi*cfoHz*cfg.timeAxisSec(t));
    Zphase = conj(phase) .* double(Z(:, :, :, t));
    Z2 = reshape(permute(Zphase, [3 1 2]), Nsc, Nr*Np);
    projectedGrad2 = C' * Z2;
    projectedGrad = permute( ...
        reshape(projectedGrad2, Nsc, Nr, Np), [2 3 1]);
    for m = 1:Nsc
        Hgrad(:, :, m, t) = projectedGrad(:, :, m) * double(P)';
    end
end

end
