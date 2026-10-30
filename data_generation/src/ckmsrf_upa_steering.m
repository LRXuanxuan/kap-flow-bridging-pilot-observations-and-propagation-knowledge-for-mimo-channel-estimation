function a = ckmsrf_upa_steering(arraySize, azRad, elRad)
%CKMSRF_UPA_STEERING Half-wavelength UPA on the y-z plane.
% The broadside is +x. azRad/elRad describe the direction from the array
% toward the transmitter/receiver or propagation object.

Ny = arraySize(1);
Nz = arraySize(2);
y = ((0:Ny-1) - (Ny-1)/2) * 0.5;
z = ((0:Nz-1) - (Nz-1)/2) * 0.5;
[Y, Z] = ndgrid(y, z);
uY = cos(elRad) * sin(azRad);
uZ = sin(elRad);
a = exp(1j * 2*pi * (Y(:)*uY + Z(:)*uZ));
a = a / sqrt(numel(a));

end
