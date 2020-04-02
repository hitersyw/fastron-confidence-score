function [t_train, t_test, l_train, l_test, Sn] = svr_benchmark(base_path, dataset, n)
    % Benchmarking the performance of SVR on a dVRK testset. 
    % 
    % Inputs:
    %    base_path - base directory for the dataset.
    %    dataset   - One of ['reachability', 'self_collision',
    %    'env_collision']
    %    n         - Number of samples in the dataset
    %
    % Outputs:
    %    t_train   - time to train the model
    %    t_test    - time to test the model
    %    l_train   - training loss
    %    l_test    - test loss
    %    Sn        - Number of support points  
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

    %% Fit Support Vector Regression on the data;
    tic()
%   Uncomment this if you want to auto-tune the parameters. 
%     svrMdl = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'KernelScale','auto',...
%                 'Solver','SMO', 'Epsilon', 0.2, ...
%                 'Standardize',false, 'verbose',0);
    svrMdl = fitrsvm(X_train,y_train,'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5, 'IterationLimit', 5000);
    t_train = toc();
    
    Sn = size(svrMdl.SupportVectors, 1);
    fprintf("Number of support points: %d\nNumber of iterations:%d\n", Sn, svrMdl.NumIterations);


    % Compute MSE Loss;
    n_test = size(X_test, 1);
    tic()
    y_pred = predict(svrMdl, X_test);
    y_pred = clip(y_pred, 0.0001); % clip the value between 0 and 1;
    t_test = toc() / n_test;
        
    eps_test = y_pred - y_test;
    l_test = eps_test' * eps_test / n_test;
   
    eps_train = clip(predict(svrMdl, X_train), 0.0001) - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);
    
    fprintf("Training MSE Loss: %.4f, Test loss: %.4f;\n", l_train, l_test);
    fprintf("Training time: %s; Test time per data point: %s\n\n", t_train, t_test);
end

