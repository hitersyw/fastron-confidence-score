function [t_train, t_test, l, Sn] = rvm_benchmark(base_path, dataset, n)

    %% Hyper-parameters;
    g = 40;
    maxIter = 5000;
    %% Load collision_score data
    input_path = base_path + 'log/%s_n%d.mat'
    [X_train, y_train, X_test, y_test] = load_dvrk(input_path, dataset, n, false);

    K = rbf(X_train, X_train, g);
    M = size(K,2);

    fprintf("Size of training set: %d; test set: %d", size(X_train,1), size(X_test,1));

    %% Define model parameters
    OPTIONS = SB2_UserOptions('iterations', maxIter,...
                              'diagnosticLevel', 2,...
                              'monitor', 10);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);

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
    K_test = rbf(X_test, X_train, g);
    y_pred = K_test(:,PARAMETER.Relevant)*w_infer(PARAMETER.Relevant); 
    % profile off;
    t_test = toc()/n_test;
    mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
    eps = y_pred - y_test;
    l = eps' * eps / size(X_test, 1);
    fprintf("MSE Loss: %.4f\n", l);
    fprintf("Training time: %s\nTest time per data point: %s\n", t_train, t_test);
    
    Sn = numel(PARAMETER.Relevant);

end
