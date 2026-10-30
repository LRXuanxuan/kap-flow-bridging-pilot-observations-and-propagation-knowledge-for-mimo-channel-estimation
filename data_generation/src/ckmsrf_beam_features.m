function features = ckmsrf_beam_features(H, Ftx, Wrx, topK, temperature)
%CKMSRF_BEAM_FEATURES Beam probabilities and top-K beam indices.

gain = abs(Wrx' * H * Ftx).^2;
txScore = max(gain, [], 1).';
rxScore = max(gain, [], 2);
txProb = localSoftmax(txScore, temperature);
rxProb = localSoftmax(rxScore, temperature);

[txSorted, txOrder] = sort(txProb, 'descend');
[rxSorted, rxOrder] = sort(rxProb, 'descend');
Ktx = min(topK, numel(txOrder));
Krx = min(topK, numel(rxOrder));

[bestGain, linearIdx] = max(gain(:));
[bestRx, bestTx] = ind2sub(size(gain), linearIdx);

features = struct();
features.txProbability = txProb;
features.rxProbability = rxProb;
features.txTopKIndex = uint16(txOrder(1:Ktx));
features.rxTopKIndex = uint16(rxOrder(1:Krx));
features.txTopKProbability = single(txSorted(1:Ktx));
features.rxTopKProbability = single(rxSorted(1:Krx));
features.bestTxIndex = uint16(bestTx);
features.bestRxIndex = uint16(bestRx);
features.bestBeamGain = single(bestGain);

end

function p = localSoftmax(score, temperature)
temperature = max(temperature, 1e-6);
score = double(score(:));
score = score / max(max(score), realmin);
scaled = (score - max(score)) / temperature;
scaled = max(scaled, -80);
p = exp(scaled);
p = single(p / max(sum(p), realmin));
end
