% Read data from files
train_data = csvread('SPECT-train.csv');
test_data = csvread('SPECT-test.csv');
train_N = size(train_data,1);
test_N = size(test_data,1);
cols = size(train_data,2);

Ts = [5 10 15 20 25 30];
%Ts = 1:40;

% Bagged decision stumps
accuracies = [];
for T = Ts
    % For each of the bags, find the split with the maximum information gain,
    % and figure out whether the class takes the same value as the splitting
    % feature or inverts it.
    splitters = zeros(1,T);
    split_sames = zeros(1,T);
    for b_num = 1:T
        % Draw a bootstrap
        bs = train_data(ceil(rand(1,train_N) * train_N),:);
        infos = arrayfun(@(f) mutual_information(bs(:,1), bs(:,f)), 2:cols);
        max_info = find(max(infos) == infos);
        max_info = max_info(1) + 1;
        splitters(b_num) = max_info;
        split_sames(b_num) = round(sum(bs(:,splitters(b_num)) == bs(:,1))/train_N);
    end

    % Predict test data labels by voting across the learned classifiers
    preds = test_data(:,splitters) == (ones(test_N,1)*split_sames);
    accuracies = [accuracies, sum(round(sum(preds,2)/T) == test_data(:,1))/test_N];
end

 plot(Ts, accuracies);


% Boosted decision stumps
accuracies = [];
for T = Ts
    D = ones(train_N,T)/train_N;
    splitters = zeros(1,T);
    alpha = zeros(1,T);
    split_sames = zeros(1,T);
    for l = 1:T
        mispreds = (train_data(:,1)*ones(1,cols-1)) ~= train_data(:,2:23);
        errors = [(~mispreds' * D(:,1)); (mispreds' * D(:,l))];
        best = find(min(errors) == errors);
        best = best(1);
        split_sames(l) = floor(best / 23);
        splitters(l) = mod(best, 23) + split_sames(l) + 1;
        alpha(l) = 1/2 * log((1 - errors(best)) / max(errors(best),eps));
        D(:,l+1) = exp(alpha(l)) / (exp(2 * alpha(l) * ((splitters(l) == train_data(:,splitters(l))) == train_data(:,1))));

        % Normalize D_{l+1}
        D(:,l+1) = D(:,l+1) / sum(D(:,l+1));
    end

    preds = (sign(((ones(test_N,1)*split_sames) == (test_data(:,splitters))-0.5) * alpha') + 1) / 2;
    accuracies = [accuracies, sum(preds == test_data(:,1)) / test_N];
end

figure(2);
plot(Ts, accuracies);