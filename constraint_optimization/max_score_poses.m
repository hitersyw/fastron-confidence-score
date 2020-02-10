% Give n best poses based on the last column of y
function [X_n_max, y_n_max] = max_score_poses(X, y, n)
    [~, index] = sortrows(y, size(y, 2), 'descend');
    top_n = index(1:n);
    % y_n_max = y(1:n, :);
    X_n_max = X(top_n, :);
    % X_n_max = X(1:n, :); 
    y_n_max = y(top_n, :);
end