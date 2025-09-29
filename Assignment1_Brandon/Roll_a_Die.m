%% ================ Part 2: Roll a die ================

% Instructions: Define a random variable "die" that represents the die, 
%               i.e., it can take six possible values, 0 - 6, with 
%               the probability of taking each value being 1/6. 
%               Then, complete the following steps. 
%                   Step 1: roll a die once, and show its value
%                   Step 2: roll a die for 10 times and make a histogram 
%                           showing the distribution (Hint: can use 
%                           function "hist()" to plot the histogram)
%                   Step 3: roll a die for 10000 times, make a histogram,
%                           and then plot an empirical cdf (Hint: can use 
%                           function "stairs()" to plot the cdf)             
% ============================================================
fprintf('===== Part 2: Roll a die ===== \n');

% ====================== YOUR CODE HERE ======================
% Step 1: roll a die once
fprintf('Roll a die ...\n');


die = randi(6);


fprintf('The value of the die: \n %d\n', die);


% Step 2: roll a die 10 times and make a histogram
fprintf('Roll a die 10 times ...\n');

die = randi([1,6],1,10);

fprintf('Showing the histogram of the die...\n');

figure;
histogram(die, 0.5:1:6.5);
xticks(1:6);
yticks(1:10);
xlabel('Die Face Number');
ylabel('Outcomes');
title('10 Die Rolls');

% Step 3: roll a die 10000 times, make a histogram, and plot the cdf
fprintf('Roll a die 10000 times ...\n');

die = randi([1,6],1,10000);

fprintf('Showing the empiral cdf of the die...\n');

figure;
histogram(die, 0.5:1:6.5);
xticks(1:6);
xlabel('Die Face Number');
ylabel('Outcomes');
title('10,000 Die Rolls');


figure;
cdfplot(die, 0.5:1:6.5);
xticks(1:6);
xlabel('Die Face');
ylabel('CDF');
title('10,000 Die Rolls');

% ============================================================