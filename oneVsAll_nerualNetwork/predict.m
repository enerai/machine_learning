function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
 
p = zeros(size(X, 1), 1);

X = [ones(m,1) X];
zTwo = X*Theta1';
zTwo = sigmoid(zTwo); %!!!!!!!!!!
zTwo = [ones(size(zTwo,1),1) zTwo];
aTwo = zTwo*Theta2';

max_in_row = max(aTwo,[],2);

for index = 1 : num_labels
	tmp = zeros(size(X, 1), 1);
	vflag = (aTwo(:,index) == max_in_row);
	tmp(vflag) = index;
	p = max(p,tmp);
end



end
