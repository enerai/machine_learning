function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));



J = sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))/m+lambda/(2*m)*sum(theta(2:end).^2);
grad = ((X'*(sigmoid(X*theta)-y))/m)+lambda/m*theta;
grad(1) = grad(1) - lambda/m*theta(1);




grad = grad(:);

end
