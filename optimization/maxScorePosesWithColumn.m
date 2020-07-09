% Give n best poses based on the last column of y
% X: the dataset
% y: the labels
% n: the top n rows dea
% col_index: the index of the columns to be sorted

function [X_n_max, y_n_max] = maxScorePosesWithColumn(X, y, n, col_index)
    [~, index] = sortrows(y, col_index, 'descend');
    top_n = index(1:n);
    % y_n_max = y(1:n, :);
    X_n_max = X(top_n, :);
    % X_n_max = X(1:n, :); 
    y_n_max = y(top_n, :);
end