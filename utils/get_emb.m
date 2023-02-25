%% get_emb
function emb = get_emb(spd_data, net)
dzdy = []; res = [] ;
len = size(spd_data, 3);
spd.data = cell(len, 1) ;
for k = 1 : len
    spd.data{k} = spd_data(:,:,k);
end
net_ = net; net_.layers(end) = [];
if strcmpi(net.NetName, 'SPD-Net')
    res = vl_myforbackward(net_, spd.data, dzdy, res) ;
elseif strcmpi(net.NetName, 'SPD-Mani-Net')
    res = vl_myforbackwardv2(net_, spd.data) ;
end

for i = numel(net_.layers):-1:1
    if strcmpi(net_.layers{i}.type, 'bfc') ||...
            strcmpi(net_.layers{i}.type, 'rec') ||...
            strcmpi(net_.layers{i}.type, 'shrinkage')
        for l = 1 : len
            emb(:,:,l) = res(i+1).x{l};
        end
        break;
    end
end

end