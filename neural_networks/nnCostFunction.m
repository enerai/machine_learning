function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



X = [ones(m,1) X];
zTwo = X*Theta1';
zTwo = sigmoid(zTwo); %!!!!!!!!!!
zTwo = [ones(size(zTwo,1),1) zTwo];
aTwo = zTwo*Theta2';

outputs = sigmoid(aTwo);


yEYEs = eye(num_labels);
yMATs = zeros(m,num_labels);
for index = 1:m
	yMATs(index,:) = yEYEs(y(index),:);
end


J = sum(sum((-log(outputs)*yMATs'-log(1-outputs)*(1-yMATs')).*eye(m)))/m;
delta = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = J + delta;

% Backpropagation 
for t = 1:m
	a_1 = X(t,:);
	z_2 = Theta1*a_1';
	a_2 = [1;sigmoid(z_2)];
	z_3 = Theta2*a_2;
	a_3 = sigmoid(z_3);
	d_3 = a_3-yMATs(t,:)';
	d_2 = Theta2'*d_3.* sigmoidGradient([1;z_2]);
	tmp_1 = d_2(2:end)*a_1;
	tmp_2 = d_3*a_2';
	Theta1_grad = Theta1_grad + tmp_1;
	Theta2_grad = Theta2_grad + tmp_2;
end

tmpT_1 = Theta1;
tmpT_1(:,1) = 0;
tmpT_2 = Theta2;
tmpT_2(:,1) = 0;

Theta1_grad = Theta1_grad + lambda*tmpT_1;
Theta2_grad = Theta2_grad + lambda*tmpT_2;

% -------------------------------------------------------------


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]/m;

end
