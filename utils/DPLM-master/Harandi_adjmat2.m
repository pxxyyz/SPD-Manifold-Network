function Adj = Harandi_adjmat2( data, labels, k )

dist = inf * ones(numel(labels));

for i = 1 : numel(labels)
    for j = i + 1 : numel(labels)
%         if(labels(i) == labels(j))
            dist(i, j) = distance_riemann(data(:, :, i), data(:, :, j));
            dist(j, i) = dist(i, j);
%         end
    end
end

Adj = zeros(numel(labels));
for i = 1 : numel(labels)
    tmpIndex = find(labels == labels(i));
    [~, sortInx] = sort(dist(tmpIndex, i));
    if (length(tmpIndex) < k)
        max_w = length(tmpIndex);
    else
        max_w = k;
    end
    Adj(i, tmpIndex(sortInx(1 : max_w))) = 1;
    
    tmpIndex = find(labels ~= labels(i));
    [~, sortInx] = sort(dist(tmpIndex, i));
    if (length(tmpIndex) < k)
        max_w = length(tmpIndex);
    else
        max_w = k;
    end
    Adj(i, tmpIndex(sortInx(1 : max_w))) = -1;
    
    %     [~,idx] = sort(dist(i,:), 'ascend');
    %     Adj(i,idx(1:k)) = 1;
end