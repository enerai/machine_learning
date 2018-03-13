function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z (vectorization version)

g = zeros(size(z));


g = (1./(1+exp(-z))).*(1-(1./(1+exp(-z))));









% =============================================================




end
