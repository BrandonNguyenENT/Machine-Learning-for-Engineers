function [beta, J_history] = gradientDescent(X, y, beta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn beta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               beta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    % ============================================================

    % Save the cost J (J = 1/(2m)*SSE) in every iteration 
    
    % ====================== YOUR CODE HERE ======================
        h = X * beta;

        % Compute gradient
        gradient = (1/m) * (X' * (h - y));

        % Update beta
        beta = beta - alpha * gradient;

        % Save cost at each iteration
        J_history(iter) = computeCost(X, y, beta);
  

    % ============================================================
end

end
