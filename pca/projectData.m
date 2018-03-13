function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%

Z = zeros(size(X, 1), K);


U_reduce = U(:,1:K);
Z = X*U_reduce;



% =============================================================

end
