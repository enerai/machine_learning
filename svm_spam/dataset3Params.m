function [C, sigma] = dataset3Params(X, y, Xval, yval)
%select best C and sigma in a set of specific numbers.
%


C = 1;
sigma = 0.3;



C_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

best_error = intmax;
best_c = 0;
best_sigma = 0;

for C = C_set
	for sigma = sigma_set
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		if (mean(double(predictions ~= yval))<best_error)
			best_c = C;
			best_sigma = sigma;
			best_error = mean(double(predictions ~= yval));
		end
	end
end

C = best_c;
sigma = best_sigma;
% =========================================================================

end
