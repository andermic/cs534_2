function [entropy] = entropy(x_draws)
% Estimate the entropy of a random variable X based on some draws from X.
% x_draws should be a column vector.

assert(size(x_draws,2) == 1);
probs = histc(x_draws, unique(x_draws)) ./ size(x_draws,1);
entropy = -sum(probs .* log(probs) / log(2));
end

