function out = ckmsrf_merge_struct(defaultStruct, userStruct)
%CKMSRF_MERGE_STRUCT Recursively merge user fields into defaults.

out = defaultStruct;
if nargin < 2 || isempty(userStruct)
    return;
end

names = fieldnames(userStruct);
for i = 1:numel(names)
    name = names{i};
    if isfield(out, name) && isstruct(out.(name)) && isstruct(userStruct.(name))
        out.(name) = ckmsrf_merge_struct(out.(name), userStruct.(name));
    else
        out.(name) = userStruct.(name);
    end
end

end
