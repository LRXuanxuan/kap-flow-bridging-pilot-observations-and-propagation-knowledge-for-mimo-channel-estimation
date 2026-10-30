function export_scene_to_hdf5(matFile, h5File)
%EXPORT_SCENE_TO_HDF5 Export pilot bank and CKM arrays to plain HDF5.

loaded = load(matFile, 'scenePackage');
p = loaded.scenePackage;
if exist(h5File, 'file')
    delete(h5File);
end

localWriteComplex(h5File, '/pilot/P', p.pilotBank.P);
localWrite(h5File, '/pilot/selected_dft_column', ...
    p.pilotBank.selectedDftColumn);
localWriteComplex(h5File, '/codebook/Ftx', p.Ftx);
localWriteComplex(h5File, '/codebook/Wrx', p.Wrx);

localWrite(h5File, '/ckm/grid_x', p.ckm.gridX);
localWrite(h5File, '/ckm/grid_y', p.ckm.gridY);
localWrite(h5File, '/ckm/grid_xy', p.ckm.gridXY);
localWrite(h5File, '/ckm/scalar_feature', p.ckm.scalarFeature);
localWrite(h5File, '/ckm/angular_delay_power', p.ckm.angularDelayPower);
localWrite(h5File, '/ckm/tx_beam_probability', p.ckm.txBeamProbability);
localWrite(h5File, '/ckm/rx_beam_probability', p.ckm.rxBeamProbability);
localWrite(h5File, '/ckm/adp_shape', p.ckm.adpShape);
localWrite(h5File, '/split/assignment', p.splitPlan.assignment);

% Write root attributes after the first dataset has created the file.
h5writeatt(h5File, '/', 'schema_version', '1.0');
h5writeatt(h5File, '/', 'scene_id', uint16(p.scene.sceneId));

end

function localWriteComplex(fileName, baseName, value)
localWrite(fileName, [baseName '_real'], real(value));
localWrite(fileName, [baseName '_imag'], imag(value));
end

function localWrite(fileName, datasetName, value)
chunk = max(size(value), 1);
if numel(chunk) > 1
    chunk(end) = 1;
end
while prod(double(chunk)) > 5e5
    [~, idx] = max(chunk);
    chunk(idx) = max(1, ceil(chunk(idx)/2));
end
h5create(fileName, datasetName, size(value), ...
    'Datatype', class(value), 'ChunkSize', chunk, 'Deflate', 4);
h5write(fileName, datasetName, value);
end
