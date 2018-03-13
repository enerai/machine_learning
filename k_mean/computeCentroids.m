function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.

% Useful variables
[m n] = size(X);


centroids = zeros(K, n);


for index_k = 1:K
	tmp = X(idx==index_k,:);
	centroids(index_k,:) = sum(tmp,1)/size(tmp,1);
end








end

