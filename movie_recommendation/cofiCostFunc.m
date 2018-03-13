function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


J = 1/2*sum(sum((Theta*X'-Y').^2.*R'))+1/2*lambda*(sum(sum(Theta.^2))+sum(sum(X.^2)));

X_grad = (Theta*X'-Y')'.*R*Theta+X*lambda;
Theta_grad = (Theta*X'-Y').*R'*X+Theta*lambda;















grad = [X_grad(:); Theta_grad(:)];

end
