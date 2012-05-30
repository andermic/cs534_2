function [mi] = mutual_information(x_draws,y_draws)
% Estimate the MI between two random variables X and Y using draws from
%  both distributions.
% x_draws and y_draws should be column vectors.

assert(size(x_draws,2) == 1);
assert(size(y_draws,2) == 1);

ux = unique(x_draws);
x_probs = histc(x_draws, ux) / size(x_draws,1);
y_given_x_entropy = x_probs' * arrayfun(@(i) entropy(y_draws(x_draws==i)), ux);

y_entropy = entropy(y_draws);
mi = y_entropy - y_given_x_entropy;
