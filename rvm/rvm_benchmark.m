function [t_train, t_test, l_train, l_test, Sn] = rvm_benchmark(base_path, dataset, n)

    %% Hyper-parameters;
    g = 1;
    maxIter = 5000;
    
    %% Load collision_score data
    input_path = sprintf(base_path + "log/%s_n%d.mat", dataset, n);
    
    [X, y] = load_dvrk3(input_path, dataset, n, false);
    
    % shuffle dataset;
    idx = randperm(n); 
    X = X(idx, :);
    y = y(idx);
    
    % normalize dataset;    
    xmax = max(X);
    xmin = min(X);
    scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
    X = scale_input(X);
    
    p_test = 0.1;
    X_test = X(1:ceil(n*p_test), :);
    y_test = y(1:ceil(n*p_test));
    X_train = X(ceil(n*p_test+1):n, :);
    y_train = y(ceil(n*p_test+1):n, :);
    fprintf("Size of training set: %d; test set: %d\n", size(X_train,1), size(X_test,1));

    % Run kernel methods
    K = rbf(X_train, X_train, g);
    M = size(K,2);

    %% Define model parameters
    OPTIONS = SB2_UserOptions('iterations', maxIter,...
                              'diagnosticLevel', 1,...
                              'monitor', 100);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.2);

    %% Fit Relevance Vector Machine (Sparse Bayesian Model) on the data;
    tic();
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
        SparseBayes('Gaussian', K, y_train, OPTIONS, SETTINGS);
    w_infer						= zeros(M,1);
    w_infer(PARAMETER.Relevant)	= PARAMETER.Value;
    t_train = toc();
    fprintf("Number of Relevance points: %d\n",size(PARAMETER.Relevant, 1));

    % Compute MSE Loss;
    n_test = size(X_test, 1);
    
    tic();
    
    % profile on;
    K_test = rbf(X_test, X_train(PARAMETER.Relevant, :), g);
    y_pred = K_test*w_infer(PARAMETER.Relevant); 
    % profile off;
    y_pred = clip(y_pred, 0.0001); % clip the value between 0 and 1;
    t_test = toc() / n_test;
    
    eps_test = y_pred - y_test;
    l_test = eps_test' * eps_test / n_test;
   
    eps_train = clip(K(:, PARAMETER.Relevant)*w_infer(PARAMETER.Relevant), 0.0001) - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);
    
    fprintf("Training MSE Loss: %.4f, Test loss: %.4f;\n", l_train, l_test);
    fprintf("Training time: %s; Test time per data point: %s\n\n", t_train, t_test);
    
    Sn = numel(PARAMETER.Relevant);

end
