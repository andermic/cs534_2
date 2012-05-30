% regression.m
% Michael Anderson

% Get training data from file
M = csvread('regression-train.csv');
x_train = M(:,1:3);
n = size(x_train,1);
x_train = cat(2,ones(n,1),x_train);
y_train = M(:,4);
epsilon = 0.0000001*ones(1,4);

% Initialization
w_init = 0;
w_batch = zeros(1,4)+w_init;
w_stochastic = zeros(1,4)+w_init;
iter = 1;
lambda = 1/n;

% Perform batch gradient descent
while true
    cur_w = w_batch(iter,:);
    yhat = x_train*cur_w';
    difs = (yhat-y_train)*ones(1,4);
    gradients = sum(difs .* x_train);
    new_w = cur_w - lambda * gradients;
    
     % Add the new w to the list of w's so far computed
    iter = iter + 1;
    w_batch(iter,:) = new_w;

    % Stop when w stops changing significantly
    if sum(abs(new_w-cur_w) > epsilon) == 0
        break;
    end
end

% Perform stochastic gradient descent
iter = 1;
while true
    % Shuffle the order that the data will be processed in
    perm = randperm(n);
    x_train = x_train(perm, :);
    y_train = y_train(perm, :);

    % Train for one epoch
    cur_w = w_stochastic(iter,:);
    in_epoch_w = cur_w;
    for iter2 = 1:n
        yhat = x_train(iter2,:)*in_epoch_w';
        dif = (yhat-y_train(iter2,:))*ones(1,4);
        gradient = dif .* x_train(iter2,:);
        in_epoch_w = in_epoch_w - lambda * gradient;
    end
    
    % Add the new w to the list of w's so far computed
    iter = iter + 1;
    w_stochastic(iter,:) = in_epoch_w;

    if (iter > 274)
        break;
    end
end

% Get test data from file
N = csvread('regression-test.csv');
x_test = N(:,1:3);
x_test = cat(2,ones(size(x_test,1),1),x_test);
y_test = N(:,4);

% Calculate batch and stochastic loss relative to test data
yhat = x_test*w_batch';
loss_batch = sum((yhat - y_test*(ones(1,size(yhat,2)))).^2);
yhat = x_test*w_stochastic';
loss_stochastic = sum((yhat - y_test*(ones(1,size(yhat,2)))).^2);

% Plot batch and stochastic loss
set(0,'DefaultAxesColorOrder',[1 0 0;0 1 0]);
semilogy(1:size(loss_batch,2), loss_batch, 1:size(loss_stochastic,2), loss_stochastic);
xlabel('Epoch');
ylabel('Loss');
title(sprintf('Batch vs. Stochastic Loss: initial w = [%d %d %d %d]', w_init, w_init, w_init, w_init));
legend('Batch', 'Stochastic');