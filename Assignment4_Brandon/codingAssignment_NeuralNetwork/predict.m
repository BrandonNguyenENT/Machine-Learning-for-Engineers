function p = predict(beta1, beta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(beta1, beta2, X) outputs the probability of the output to 
%   1, given input X and trained weights of a neural network (beta1, beta2)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. 
%
%


m = size(X, 1);


p = zeros(m, 1);


X_bias = [ones(m, 1) X];

z2 = X_bias * beta1';           
a2 = 1 ./ (1 + exp(-z2));       

a2_bias = [ones(m, 1) a2];


z3 = a2_bias * beta2';         
a3 = 1 ./ (1 + exp(-z3));      

p = a3 >= 0.5;




% =========================================================================


end
