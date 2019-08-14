% Logistics Regression training procedure;
function [p] = trainLR(X, y)
    y(y == -1) = 2; % mnrfit requires labels to be positive numbers;
    B = mnrfit(X, y);
    p = B(:, :); % return only the positive prob;
end
