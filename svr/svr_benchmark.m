
function [t_train, t_test, l, Sn] = svr_benchmark(base_path, dataset, n)

    %% Load collision_score data
    input_path = sprintf(base_path + "log/%s_n%d.mat", dataset, n);
    [X_train, y_train, X_test, y_test] = load_dvrk(input_path, dataset, n, false);

    fprintf("Size of training set: %d; test set: %d\n", size(X_train,1), size(X_test,1));

    %% Fit Support Vector Regression on the data;
    tic()
    svrMdl = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'KernelScale','auto',...
                'Solver','SMO', 'Epsilon', 0.2, ...
                'Standardize',false, 'verbose',0);
    t_train = toc();
    
    Sn = size(svrMdl.SupportVectors, 1);
    fprintf("Number of support points: %d\nNumber of iterations:%d\n", Sn, svrMdl.NumIterations);


    % Compute MSE Loss;
    n_test = size(X_test, 1);
    tic()
    y_pred = predict(svrMdl, X_test);
    t_test = toc() / n_test;
    mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
    eps = y_pred - y_test;
    l = eps' * eps / n_test;
    fprintf("MSE Loss: %.4f\n", l);
    fprintf("Training time: %s\nTest time per data point: %s\n", t_train, t_test);
end

