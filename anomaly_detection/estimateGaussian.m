function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X\

% Useful variables
[m, n] = size(X);


mu = zeros(n, 1);
sigma2 = zeros(n, 1);

mu=mean(X)';
sigma2 = var([X;mu'])';








% =============================================================


end
