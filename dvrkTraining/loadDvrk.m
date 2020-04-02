function [X_train, y_train, X_test, y_test] = load_dvrk2(input_path, dataset, n, show_dist, shuffle, lb)
    % Load the dvrk dataset from base directory
    % 
    % Inputs:
    %    input_path - directory containing the dataset
    %    dataset   - One of ['reachability', 'self_collision',
    %    'env_collision']
    %    n         - Number of samples in the dataset
    %    show_dist - display distribution of samples
    %    shuffle   - whether to shuffle the dataset
    %    lb        - Lower bound for including the dataset
    %
    % Outputs:
    %    X_train   - Training data
    %    y_train   - Training label
    %    X_test    - Test data
    %    y_test    - Test label 
    score_dict = load(sprintf(input_path, dataset, n));
    score = getfield(score_dict, dataset);
    X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
    y = score(:, 5);
    
    % Set lower bound to filter
    if nargin < 6, lb = 0; end
    
    nz = y >= lb;
    X = X(nz, :);
    y = y(nz);
    
    n = size(X, 1);
    input_type = "Score from 0 to 1";

    p_test = 0.1;
  
    if shuffle
        % shuffle the dataset;
        idx = randperm(n); 
        X = X(idx, :);
        y = y(idx);
    end
    X_test = X(1:ceil(n*p_test), :);
    y_test = y(1:ceil(n*p_test));

    X_train = X(ceil(n*p_test+1):n, :);
    y_train = y(ceil(n*p_test+1):n);
    
    if show_dist
        close all;
        figure; clf;
        histogram(y, 10,'Normalization','probability');
        title(sprintf('%s, n:%d', strrep(dataset, '_', ' '), n));
    end
end