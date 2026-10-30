function export_shard_to_hdf5(matFile, h5File)
%EXPORT_SHARD_TO_HDF5 Export the SRF-relevant shard fields to plain HDF5.
%
% Complex arrays are split into *_real and *_imag datasets so that Python
% can read them without MATLAB object/reference decoding.

loaded = load(matFile, 'shard');
shard = loaded.shard;
if exist(h5File, 'file')
    delete(h5File);
end

localWriteComplex(h5File, '/H', shard.H_true);
localWriteComplex(h5File, '/Y', shard.Y_pilot);
localWrite(h5File, '/channel_scale_lin', shard.channelScaleLin);

localWrite(h5File, '/state/position_xyz', shard.state.positionXYZ);
localWrite(h5File, '/state/velocity_xyz', shard.state.velocityXYZ);
localWrite(h5File, '/state/ue_orientation_deg', shard.state.ueOrientationDeg);
localWrite(h5File, '/state/time_sec', shard.state.timeSec);
localWrite(h5File, '/state/block_id', shard.state.blockId);

localWrite(h5File, '/context/observed_position_xyz', ...
    shard.context.observedPositionXYZ);
localWrite(h5File, '/context/grid_indices', shard.context.gridIndices);
localWrite(h5File, '/context/grid_weights', shard.context.gridWeights);
localWrite(h5File, '/context/scalar_feature', shard.context.scalarFeature);
localWrite(h5File, '/context/angular_delay_power', ...
    shard.context.angularDelayPower);
localWrite(h5File, '/context/tx_beam_probability', ...
    shard.context.txBeamProbability);
localWrite(h5File, '/context/rx_beam_probability', ...
    shard.context.rxBeamProbability);

localWrite(h5File, '/truth/path_id', shard.truth.pathId);
localWrite(h5File, '/truth/path_mask', uint8(shard.truth.pathMask));
localWriteComplex(h5File, '/truth/path_gain', shard.truth.pathGain);
localWrite(h5File, '/truth/path_power_lin', shard.truth.pathPowerLin);
localWrite(h5File, '/truth/delay_sec', shard.truth.delaySec);
localWrite(h5File, '/truth/aod_az_rad', shard.truth.aodAzRad);
localWrite(h5File, '/truth/aod_el_rad', shard.truth.aodElRad);
localWrite(h5File, '/truth/aoa_az_rad', shard.truth.aoaAzRad);
localWrite(h5File, '/truth/aoa_el_rad', shard.truth.aoaElRad);
localWrite(h5File, '/truth/doppler_hz', shard.truth.dopplerHz);

localWrite(h5File, '/observation/pilot_id', shard.observation.pilotId);
localWrite(h5File, '/observation/snr_db', shard.observation.snrDb);
localWrite(h5File, '/observation/cfo_hz', shard.observation.cfoHz);
localWrite(h5File, '/observation/noise_var', shard.observation.noiseVar);
localWrite(h5File, '/sample_seed', shard.sampleSeed);

% Write root attributes after the first dataset has created the file.
h5writeatt(h5File, '/', 'schema_version', '1.0');
h5writeatt(h5File, '/', 'dimension_order_H', shard.metadata.dimensionOrderH);
h5writeatt(h5File, '/', 'dimension_order_Y', shard.metadata.dimensionOrderY);
h5writeatt(h5File, '/', 'split_name', shard.metadata.splitName);
h5writeatt(h5File, '/', 'scene_id', uint16(shard.metadata.sceneId));

end

function localWriteComplex(fileName, baseName, value)
if isempty(value)
    return;
end
localWrite(fileName, [baseName '_real'], real(value));
localWrite(fileName, [baseName '_imag'], imag(value));
end

function localWrite(fileName, datasetName, value)
if isempty(value)
    return;
end
chunk = localChunkSize(size(value));
h5create(fileName, datasetName, size(value), ...
    'Datatype', class(value), 'ChunkSize', chunk, 'Deflate', 4);
h5write(fileName, datasetName, value);
end

function chunk = localChunkSize(dataSize)
chunk = max(dataSize, 1);
if numel(chunk) > 1
    chunk(end) = 1;
end
while prod(double(chunk)) > 5e5
    [~, idx] = max(chunk);
    chunk(idx) = max(1, ceil(chunk(idx)/2));
end
end
