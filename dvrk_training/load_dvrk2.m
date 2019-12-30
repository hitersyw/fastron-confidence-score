function [X_train, y_train, X_test, y_test] = load_dvrk2(input_path, dataset, n, show_dist, lb)
    
    score_dict = load(sprintf(input_path, dataset, n));
    score = getfield(score_dict, dataset);
    X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
    y = score(:, 5);
    
    % Set lower bound to filter
    if nargin < 5, lb = 0; end
    
    nz = y >= lb;
    X = X(nz, :);
    y = y(nz);
    
    n = size(X, 1);
    input_type = "Score from 0 to 1";

    p_train = 0.95;
    idx = randperm(n); % shuffle the dataset;
    X = X(idx, :);
    X_train = X(1:ceil(n*p_train), :);
    y_train = y(1:ceil(n*p_train));

    X_test = X(ceil(n*p_train+1):n, :);
    y_test = y(ceil(n*p_train+1):n);
    
    if show_dist
        close all;
        figure; clf;
        histogram(y, 10,'Normalization','probability');
        title(sprintf('%s, n:%d', strrep(dataset, '_', ' '), n));
    end
end