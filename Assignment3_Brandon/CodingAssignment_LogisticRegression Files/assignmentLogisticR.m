%% SDSU Machine Learning Course (CompE510/EE600/CompE596)
%% Programming Assignment:  Logistic regression 
%
%  Dataset comes from: 
%   http://networkrepository.com/pima-indians-diabetes.php
% 
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  logistic regression assignment. 
%
%  You will need to complete the following functions in this 
%  assignment
%
%     loadData.m
%     featureNormalize.m
%     gradientAscent.m
%     likelihoodFunction.m
%     evaluateAccuracy.m 
%     predict.m
%     sigmoid.m
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
% Initialization
clear ; close all; clc

%% ================ Part 1: Data Preprocessing ================
fprintf('===== Part 1: Data Preprocessing ===== \n');

% Instructions: The following code loads data into matlab, splits the 
%               data into two sets, and performs feature normalization. 
%               You will need to complete code in loadData.m, and 
%               featureNormalize.m
% ============================================================

fprintf('Loading data ...\n');
% ====================== YOUR CODE HERE ======================
% Step 1: Load data
[X_train, y_train, X_test, y_test] = loadData();

% ============================================================

[num_train, m] = size(X_train); 
                                
             
fprintf('First 10 examples from the training dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n',...
    [X_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');


fprintf('Normalizing Features ...\n');
% ====================== YOUR CODE HERE ======================
% Step 2: Normalize the features. 
[Xn_train, mu, sigma] = featureNormalize(X_train);

% ============================================================

fprintf('First 10 examples from the training dataset after normalization: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n',...
    [Xn_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');


Xn_train = [ones(num_train, 1) Xn_train];


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ========== Part 2: Maximum Likelihood & Gradient Ascent =============

fprintf('===== Part 2: Maximum Likelihood & Gradient Ascent ===== \n');

% Instructions: The following code applies gradient ascent to 
%               estimate the parameters in a logistic regression 
%               model based on the idea of maximum likelihood estimation. 
%               You should complete code in gradientAscent.m,
%               likelihoodFunction.m
%
%               Try running gradient ascent with 
%               different values of alpha and see which one gives
%               you the best result.
% ============================================================

% ====================== YOUR CODE HERE ======================
% Step 1: Configure the hyper-parameters, including
alpha = 0.01;
num_iters = 400;    
% ============================================================


beta = zeros(size(Xn_train, 2), 1);

% ====================== YOUR CODE HERE ======================
% Step 2: Run Gradient Ascent
[beta, l_history] = gradientAscent(Xn_train, y_train, beta, alpha, num_iters);
% ============================================================


figure;
plot(1:numel(l_history), l_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Log Likelihood l');
title('Gradient Ascent Convergence');


fprintf('beta computed from gradient ascent: \n');
fprintf(' %f \n', beta);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 3: Evaluate performance =================
fprintf('===== Part 3: Evaluate Performance ===== \n');

% Instructions: The following code evaluates the performance of
%               the trained logistic regression model. You should 
%               complete code in evaluateAccuracy.m, predict.m, and sigmoid.m
% ============================================================


% ====================== YOUR CODE HERE ======================
% Step 1: Evaluate the performance of the trained model
% Hint: The testing set also needs to be normlized first
Xn_test = (X_test - mu) ./ sigma;
Xn_test = [ones(size(Xn_test, 1), 1) Xn_test];
accuracy = evaluateAccuracy(beta, Xn_test, y_test);

% ============================================================

fprintf('Accuracy:\n %f\n', accuracy);
fprintf('\n');

% ====================== YOUR CODE HERE ======================
% Step 2: Given a new input x = [3, 100, 79, 19, 100, 36, 0.8, 30], predict the output
x_new = [3, 100, 79, 19, 100, 36, 0.8, 30];
x_new = (x_new - mu) ./ sigma;
x_new = [1, x_new];

y_new = predict(beta, x_new);

% ============================================================
% display the predicted output
fprintf('Predicted output:\n %d\n', y_new);
