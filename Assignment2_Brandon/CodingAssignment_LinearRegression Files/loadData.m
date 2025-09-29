function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
%   and splits the dataset into two sets: training set and testing set
%
%   We only use three features as the input X: 
%       X2=the house age (unit:year)
%       X3=the distance to the nearest MRT station (unit:degree)
%       X4=the number of convenience stores in the living circle on foot (integer)
%   The output y is:
%       y=house price of unit area (10000 New Taiwan Dollar/Ping, where 
%         Ping is a local unit, 1 Ping = 3.3 meter squared)

 % ====================== YOUR CODE HERE ======================
    % Instructions: Import spreadsheets data and extract the columns
    % corresponding to X2, X3, X4 and store them as X. Extract the last
    % column and store it as y. 
    %
    % Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    %
    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    %

    % Load Provided SPREADSHEET File
    data = readtable('housePriceData.xlsx', 'VariableNamingRule', 'preserve');

    % Extract relevant columns: X2, X3, X4
    % Dataset Columns: 
    X = table2array(data(:, {'X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores'}));
    y = table2array(data(:, 'Y house price of unit area'));


    m = size(X,1);
    idx = randperm(m);
    train_size = round(0.7 * m);

    X_train = X(idx(1:train_size), :);
    y_train = y(idx(1:train_size), :);

    X_test  = X(idx(train_size+1:end), :);
    y_test  = y(idx(train_size+1:end), :);

    
 % ============================================================   
    
end