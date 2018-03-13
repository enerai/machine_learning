function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example

% Set K
K = size(centroids, 1);


idx = zeros(size(X,1), 1);


num_X = size(X,1);
for index_x = 1:num_X
	bestValue = intmax;
	for index_k = 1:K
		tmp = sum((X(index_x,:)-centroids(index_k,:)).^2);
		if (tmp < bestValue)
			idx(index_x) = index_k;
			bestValue = tmp;
		end
	end
end

% values_xk = zeros(num_X,K);




end

