function C = ckmsrf_cfo_matrix(numSubcarriers, normalizedCfo)
%CKMSRF_CFO_MATRIX OFDM inter-carrier interference matrix.
% C(k,m) maps the pre-CFO signal on subcarrier m to observed subcarrier k.

n = 0:numSubcarriers-1;
C = complex(zeros(numSubcarriers), zeros(numSubcarriers));
for k = 1:numSubcarriers
    for m = 1:numSubcarriers
        q = normalizedCfo + (m-1) - (k-1);
        C(k, m) = mean(exp(1j * 2*pi * q * n / numSubcarriers));
    end
end

end
