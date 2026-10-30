function Y = ckmsrf_apply_operator(H, P, cfoHz, cfg)
%CKMSRF_APPLY_OPERATOR Apply finite pilots and optional CFO/ICI.
%
% H: Nr x Nt x Nsc x T
% P: Nt x Np
% Y: Nr x Np x Nsc x T

Nr = cfg.numRxAntennas;
Np = size(P, 2);
Nsc = cfg.numSubcarriers;
T = cfg.numSnapshots;
if ~isequal(size(H, 1), Nr) || ~isequal(size(H, 2), cfg.numTxAntennas)
    error('CKMSRF:Operator', 'H has incompatible antenna dimensions.');
end
if size(H, 3) ~= Nsc || size(H, 4) ~= T
    error('CKMSRF:Operator', 'H has incompatible subcarrier/snapshot dimensions.');
end

if cfg.cfo.enable
    C = ckmsrf_cfo_matrix(Nsc, cfoHz / cfg.subcarrierSpacingHz);
else
    C = eye(Nsc);
end

Y = complex(zeros(Nr, Np, Nsc, T), zeros(Nr, Np, Nsc, T));
for t = 1:T
    projected = complex(zeros(Nr, Np, Nsc), zeros(Nr, Np, Nsc));
    for m = 1:Nsc
        projected(:, :, m) = double(H(:, :, m, t)) * double(P);
    end
    projected2 = reshape(permute(projected, [3 1 2]), Nsc, Nr*Np);
    observed2 = C * projected2;
    observed = permute(reshape(observed2, Nsc, Nr, Np), [2 3 1]);
    phase = exp(1j * 2*pi*cfoHz*cfg.timeAxisSec(t));
    Y(:, :, :, t) = phase .* observed;
end

end
