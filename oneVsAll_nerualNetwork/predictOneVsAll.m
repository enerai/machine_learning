function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. 

m = size(X, 1);
num_labels = size(all_theta, 1);


p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


res = X*all_theta';
max_in_row = max(res,[],2);

for index = 1 : num_labels
	tmp = zeros(size(X, 1), 1);
	vflag = (res(:,index) == max_in_row);
	tmp(vflag) = index;
	p = max(p,tmp);
end


end
