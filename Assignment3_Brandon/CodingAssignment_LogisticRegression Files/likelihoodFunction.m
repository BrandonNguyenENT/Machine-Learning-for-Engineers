function [l, grad] = likelihoodFunction(beta, X, y)
%LIKELIHOODFUNCTION Computes log likelihood using beta as the parameter 
%   for logistic regression and the gradient of the log likelihood function
%   w.r.t. to the parameters..

m = length(y);
h = sigmoid(X * beta);   % predicted probabilities

% ====================== YOUR CODE HERE =====================
% Instructions: Compute the log-likelihood (l) of a particular choice of beta
%               averaged over all training examples, i.e., l = 1/num_train * (...) 
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the log-likelihood function w.r.t. each 
%               parameter in beta
%
% Note: grad should have the same dimensions as beta
%

l = (1 / m) * sum(y .* log(h) + (1 - y) .* log(1 - h));

grad = (1 / m) * (X' * (y - h));

end
