function error = evaluateAccuracy(beta, X, y)
%EVALUATEACCURACY calculates the average prediction error of the learned 
%linear regression model using the testing data 

m = length(y); % number of testing examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the prediction error (mean squared error) using the
%               testing set.
%

y_pred = X * beta;                 % predictions
sq_errors = (y_pred - y).^2;       % squared differences
error = mean(sq_errors);           % average error (MSE)

% ============================================================

end