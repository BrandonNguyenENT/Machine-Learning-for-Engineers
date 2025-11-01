function W = findPCs(X, K)
% This function finds the first K principle components of X

% ====================== YOUR CODE HERE ======================
% Instructions: First, compute the estimated variance of X, denoted as S
%               Second, compute eigenvalues and corresponding 
%               eigenvectors of S
%               Third, sort the eigenvalues in a descend order
%               Four, find the top K principle components with the largest
%               eigenvalues
% Hints:        try functions cov(), eig() 
%

% Step 1: Compute the covariance matrix S
S = cov(X);

% Step 2: Compute eigenvalues and eigenvectors of S
[V, D] = eig(S);

% Step 3: Sort the eigenvalues (and eigenvectors) in descending order
[~, idx] = sort(diag(D), 'descend');
V_sorted = V(:, idx);

% Step 4: Select the top K principal components
W = V_sorted(:, 1:K);

% ============================================================
end