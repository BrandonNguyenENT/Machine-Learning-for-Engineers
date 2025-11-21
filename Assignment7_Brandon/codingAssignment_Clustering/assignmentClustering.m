%% SDSU Machine Learning Course (CompE510/EE600/CompE596)
%% Programming Assignment:  K-Means Clustering 
%
%  Instructions
%  ------------
% 
%  Dataset comes from: 
%   https://leafyplace.com/types-of-birds/
% 
%
% 
%  This file contains code that helps you get started on the
%  k-means clustering assignment. 
%
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  the number of clusters).


% Initialization
clear ; close all; clc

%% =========== Part 1: Load Image ============================

fprintf('===== Part 1: Load Image ===== \n');

% Instructions: The following code loads and display image bluebird.jped
%               Hint: you can use functions "imread()" and "imshow()"
% ============================================================

% ====================== YOUR CODE HERE ======================
% Step 1: read the image and store it in "img"
img = imread('bluebird.jpeg');

% Step 2: display the image
imshow(img);

% ============================================================

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Cluster Pixels with K-means =============
fprintf('===== Part 2: Cluster Pixels with K-means ===== \n');

% Instructions: The following code cluster the data points (pixels) into K 
%               clusters. Tune the value of K to see its impact.  
%
%               Procedure: 
%               First initilize the centroids
%               Then iterate following steps until convergence (iterate for a certain number of iterations)
%                   a: Assign each pixel into the cluster with centroid closest to it
%                   b: Update the centroids 
% ============================================================


% ====================== YOUR CODE HERE ======================
% Step 1: Configure the number of clusters
K = 30;

% Step 2: scale the pixel values so that they fall within [0,1]
img = imread('bluebird.jpeg');     
img = im2double(img);             

% Step 3: convert the image as a 2d matrix 
% Hint: if an image is of dimension (n,m,c), then reshape it 
%       to a matrix of dimension (n*m,c)

[n, m, c] = size(img);
X = reshape(img, n*m, c);

% Step 4: initilize the cluster centroids

rand_idx = randperm(size(X,1), K);
centroids = X(rand_idx, :);

% Step 5: iterate to cluster the pixels
num_iterations = 10;
SSE = zeros(num_iterations,1);
for iter = 1:num_iterations
% Step 5a: assign each pixel into a cluster

distances = pdist2(X, centroids);            
[~, idx] = min(distances, [], 2);       

% Step 5b: recalculate the centroids

    for k = 1:K
        centroids(k,:) = mean(X(idx == k,:), 1);
    end

    sse_iter = 0;
    for k = 1:K
        sse_iter = sse_iter + sum(sum((X(idx == k,:) - centroids(k,:)).^2));
    end
    SSE(iter) = sse_iter;

end

% ============================================================
% Step 6: Plot a graph that shows the change of SSE (Sum of Squared Error) as a way
% to check convergence
figure;
plot(1:num_iterations, SSE, '-o');
xlabel('Iteration');
ylabel('Sum of Squared Error (SSE)');
title('K-means Clustering Convergence');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Construct the Compressed Image ================
fprintf('===== Part 3: Construct the Compressed Image ===== \n');

% Instructions: The following code construct and display the compressed image.
%               Hint: pixels within the same cluster have the same value as
%               the cluster centroid.  
%
% 
% ====================== YOUR CODE HERE ======================
X_compressed = centroids(idx, :);

img_compressed = reshape(X_compressed, n, m, c);


% ============================================================

% store the compressed image 
imwrite(img_compressed, 'bluebird_compressed.jpeg');
% display the compressed image
fprintf('Compression is done. \n');
figure;
imshow(img_compressed)
title(sprintf('Compressed Image (K = %d)', K));






