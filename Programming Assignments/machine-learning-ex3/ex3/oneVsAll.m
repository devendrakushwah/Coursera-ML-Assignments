function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


initial_theta = zeros(n + 1,1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1:num_labels,
  [theta]  = fmincg(@(initial_theta)lrCostFunction(initial_theta, X, (y == i), lambda),initial_theta,options);
  all_theta(i,:) = theta;
end;

end