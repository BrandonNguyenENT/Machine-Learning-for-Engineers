%% ================ Part 3: Plot a normal distribution ================

% Instructions: Generate 10000 random samples from the normal distribution 
%               with mean = 1 and variance = 4 (Hint: use function "rand()").
%               Then complete the following:
%                   Step 1: Make a histogram to show the distribution.
%                   Step 2: Fit a probability density function (normal distribution) 
%                           to the data and plot the pdf superimposed over a histogram 
%                           of the data (Hint: use function "fitdist()")             
% ============================================================

fprintf('===== Part 3: Plot a normal distribution ===== \n');

% ====================== YOUR CODE HERE ======================
% Step 1: Make a histogram to show the distribution.

samples = 1 + 2*randn(1, 10000);

fprintf('Showing the histogram of the data ...\n');

figure;
histogram(samples,50,'Normalization','pdf');
xlabel('Values');
ylabel('Density');
title('Normal Distribution Histogram');

% Step 2: Fit a probability density function to the data and plot the pdf 
%     superimposed over a histogram of the data (Hint: use function "fitdist()")

fprintf('Showing the pdf created by fitting a normal distribution to the data ...\n');

fnd = fitdist(samples','Normal');              
x = linspace(min(samples), max(samples), 100);  
y = pdf(fnd, x);
hold on;
plot(x,y,'r','LineWidth',2);               
legend('Histogram','Fitted Normal PDF');
hold off;

% ============================================================
