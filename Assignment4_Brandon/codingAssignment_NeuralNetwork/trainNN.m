function [beta1, beta2, J_history] = trainNN(X, y, beta1, beta2, alpha, num_epochs);
%TRAINNN train the neural network model using backpropagation algorithm. It
%updates the weights, beta1 and beta2 using the training examples. It also
%generates the cost computed after each epoch. 

% useful values
[n, ~] = size(X); % n is number of training examples
num_hidden = size(beta1, 1);  % number of hidden units
num_output = size(beta2, 1);  % number of output units


J_history = zeros(num_epochs,1); % stores value of the cost function J at each iteration

for epoch = 1:num_epochs
% for each training example, do the following
    Jd = 0;
    for d = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the weights beta1 and
    %               beta2. The key steps are indicated as follows
    %
    %
    
 
        %% Step 1: forward propagate to generate the network output
        a1 = [1; X(d,:)'];                 
        z2 = beta1 * a1;
        a2 = [1; sigmoid(z2)];           
        z3 = beta2 * a2;
        a3 = sigmoid(z3);                 

        
                
        %% Step 2: for each output unit, calculate its error term
        % Recall that the number of output units is num_output
        delta3 = a3 - y(d);
        
        
        %% Step 3: for each hidden unit, calculate its error term
        % Recall that number of hidden units is num_hidden+1
        delta2 = (beta2(:,2:end)' * delta3) .* a2(2:end) .* (1 - a2(2:end));
        
        
        

        %% Step 4: update the weights using the error terms
        grad_beta2 = delta3 * a2';
        grad_beta1 = delta2 * a1';

        beta2 = beta2 - alpha * grad_beta2;
        beta1 = beta1 - alpha * grad_beta1;

        
        %% calculate the cost (Jd = SSE) per epoch
        Jd = Jd + sum(delta3.^2);
    end
    J_history(epoch) = Jd/(2*n);
end