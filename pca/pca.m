function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X

% Useful values
[m, n] = size(X);


U = zeros(n);
S = zeros(n);


Sigma = 1/m*X'*X;
[U,S,] = svd(Sigma);






end
