function [t_train, t_test, l, model] = mlp_benchmark(input_path, dataset, n)

    %% Load dataset
    [X_train, y_train, X_test, y_test] = load_dvrk(input_path, dataset, n, false);
    
    fprintf("Size of training set: %d; test set: %d\n", size(X_train,1), size(X_test,1));

    %% Fit MLP for regression on it;
    tic();
    h = [64,64];            % two hidden layers with 64 and 64 neurons
    lambda = 0.0001;        % Regularized Loss; 
    [model, L] = mlpRegSigmoid(X_train',y_train',h,lambda);
    % p_mlp_clipped = max(0, min(1, p_mlp)); % clip the values between [0,1];
    t_train = toc();

    % Compute MSE Loss;
    tic()
    n_test = size(X_test, 1);
    y_pred = mlpRegPred(model, X_test')';
    t_test = toc() / n_test;
    mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
    eps = y_pred - y_test;
    l = eps' * eps / n_test;
    fprintf("MSE Loss: %.4f\n", l);
    fprintf("Training time: %s\nTest time per data point: %s\n", t_train, t_test);
end

