% perceptron.m
% Michael Anderson

% Get data from file
M = csvread('twogaussian.csv');
n = size(M,1);
x = cat(2,ones(n,1),M(:,2:3));
y = M(:,1);

% Initialize
w = [0 0 0];
lambda = 1;
epsilon = 0.00000001 * ones(1,3);

% Perform batch perceptron
for iter = 1:Inf
    delta = [0 0 0];
    misses(iter) = 0;
    for m = 1:n
        u(m) = w * x(m,:)';
        if y(m) * u(m) <= 0;
            delta = delta - y(m) * x(m,:);
            misses(iter) = misses(iter) + 1;
        end
    end
    delta = delta / n;
    w = w - lambda * delta;
    if sum(abs(delta) >= epsilon) == 0
        break;
    end
end

% Plot number of misclassifications as a function of number of epochs
plot(1:iter, misses);
title('Number of Misclassifications at Each Epoch');
xlabel('Epoch');
ylabel('Number of Misclassification');

% Plot data and learned decision boundary. Comment out this block to plot
% misclassifications.
dec_x = -4:0.01:8;
dec_y = -(w(1) + dec_x*w(2))/w(3);
plot(x(1:98,2), x(1:98,3), 'x', x(99:n,2), x(99:n,3), 'o', dec_x,dec_y);
title('Perceptron Test Data and Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Positive', 'Negative', 'Decision Boundary');