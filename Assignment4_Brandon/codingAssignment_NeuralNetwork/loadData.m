function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   http://networkrepository.com/pima-indians-diabetes.php
%   and splits the dataset into two sets: training set and testing set
%

 % ====================== YOUR CODE HERE ======================
    % Instructions: Import spreadsheets data, extract the first
    % 8 columns and store them as X. Extract the last column and 
    % store it as y. 
    %
    % Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    %
    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    %

    data = readtable('pima-indians-diabetes.csv');

    
    datatable = table2array(data);

    
    X = datatable(:, 1:8);
    y = datatable(:, 9);

    
    m = size(X, 1);
    rand_indices = randperm(m);
    split_index = floor(0.7 * m);

    X_train = X(rand_indices(1:split_index), :);
    y_train = y(rand_indices(1:split_index), :);

    X_test = X(rand_indices(split_index+1:end), :);
    y_test = y(rand_indices(split_index+1:end), :);
    
    


% ============================================================
end