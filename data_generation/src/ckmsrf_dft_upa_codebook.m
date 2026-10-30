function F = ckmsrf_dft_upa_codebook(arraySize)
%CKMSRF_DFT_UPA_CODEBOOK Normalized separable DFT codebook for a UPA.
Fv = localDft(arraySize(1));
Fh = localDft(arraySize(2));
F = kron(Fh, Fv);
end

function F = localDft(N)
n = (0:N-1).';
k = 0:N-1;
F = exp(-1j * 2*pi * n * k / N) / sqrt(N);
end
