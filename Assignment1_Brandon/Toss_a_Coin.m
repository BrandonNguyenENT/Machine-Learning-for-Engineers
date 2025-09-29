%% ================ Part 1: Toss a coin ===================

% Instructions: Define a random variable called "coin" that represents the coin, 
%               i.e., it can take two possible values, 0 (tail) 
%               and 1 (head), with the probability of taking each 
%               value being 0.5. 
%               Then, complete the following steps. 
%                   Step 1: toss a coin once, and show the value of the
%                           coin
%                   Step 2: toss a coin for 10 times, count the 
%                           number of heads. Store the number of 
%                           heads in variable "c" and print its value
% ============================================================
fprintf('===== Part 1: Toss a coin ===== \n');


% ====================== YOUR CODE HERE ======================
% Step 1: toss a coin, and show the value of the coin
fprintf('Toss a coin ...\n');

if rand() <= 0.5
    coin = 1; % Head
else
    coin = 0; % Tail
end

fprintf('The value of the coin: \n %d\n', coin);

% Step 2: toss a coin for 10 times and count the number of heads
% Store the number of heads in variable "c" and print its value
fprintf('Toss a coin for 10 times ...\n');

coin = rand(1,10) <= 0.5;  % Generates 10 coin tosses
c = sum(coin);           % Total Number of Heads

fprintf('The number of heads: \n %d\n', c);
% ============================================================

