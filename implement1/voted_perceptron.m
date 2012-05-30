% voted_perceptron.m
% Michael Anderson

% Get data from file
M = csvread('iris-twoclass.csv');
N = size(M,1);
x_orig = cat(2,ones(N,1),M(:,2:3));
y = M(:,1);

% Initialize
w = [0 0 0];
c = 0;
n = 1;

% Perform voted perceptron
for epoch = 1:100
    % Randomly shuffle data at the beginning of each epoch
    ordering = randperm(N);
    x = x_orig(ordering,:);
    y = y(ordering);
    
    for i = 1:N
        u = w(n,:) * x(i,:)';
        if (y(i) * u) <= 0
            w(n+1,:) = w(n,:) + y(i)*x(i,:);
            c(n+1) = 0;
            n = n + 1;
        else
            c(n) = c(n) + 1;
        end
    end
end

% Classify some feature values in a fine grid to determine the decision boundary
for i = 1:1:70
    for j = 0.1:0.1:3
        if sum(c .* ((([1 i/10 j] * w') > 0) * 2 - 1)) < 0
            break;
        end
    end
    dec_y(i) = j;
end

% Plot results
plot(x_orig(1:100,2), x_orig(1:100,3), 'x', x_orig(101:N,2), x_orig(101:N,3), 'o', 0.1:0.1:7, dec_y);
title('Voted Perceptron Test Data and Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Positive', 'Negative', 'Decision Boundary', 'Location', 'SouthEast');