function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

		h = sigmoid(X * theta);
		theta(1) = 0;
		J = -1 * ((y' * log(h)) + ((1 .- y)' * log(1 - h))) / m;

		J = J + (lambda * sum(theta .^ 2) / (2 * m));

		grad = (X' * (h - y)) ./ m;
		theta = theta .* (lambda/m);
		grad = grad + theta;

grad = grad(:);

end
