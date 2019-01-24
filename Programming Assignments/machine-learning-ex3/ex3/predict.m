function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(size(X,1),1) X];

a1 = sigmoid(X * Theta1');
a1 = [ones(size(X,1),1) a1];

a2 = sigmoid(a1 * Theta2');

[temp, p] = max(a2,[],2);

end