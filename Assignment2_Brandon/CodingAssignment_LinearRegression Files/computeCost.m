function J = computeCost(X, y, beta)
%COMPUTECOST computes the cost of using beta as the
%   parameter for linear regression to fit the data points in X and y

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of beta
%               You should set J to the cost.

    % Number of training examples
    m = length(y);

    % Predictions of hypothesis on all m examples
    h = X * beta;

    % Squared errors
    errors = h - y;

    % Cost function
    J = (1 / (2 * m)) * sum(errors .^ 2);


% =========================================================================

end
