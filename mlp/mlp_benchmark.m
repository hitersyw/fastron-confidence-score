function [t_train, t_test, l, model] = mlp_benchmark(dataset, n)

    %% Load dataset
    score_dict = load(sprintf('/home/jamesdi1993/workspace/arclab/fastron_experimental/fastron_vrep/constraint_analysis/log/%s_n%d.mat', dataset, n));
    score = getfield(score_dict, dataset);
    X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
    y = score(:, 5);
    n = size(X, 1);
    input_type = "Score from 0 to 1";

    p_train = 0.8;
    idx = randperm(n); % shuffle the dataset;
    X = X(idx, :);
    X_train = X(1:ceil(n*p_train), :);
    y_train = y(1:ceil(n*p_train));

    X_test = X(ceil(n*p_train+1):n, :);
    y_test = y(ceil(n*p_train+1):n);

    fprintf("Size of training set: %d; test set: %d\n", size(X_train,1), size(X_test,1));

    %% Fit MLP for regression on it;
    tic();
    h = [64,64];            % two hidden layers with 64 and 64 neurons
    lambda = 0.0001;        % Regularized Loss; 
    [model, L] = mlpReg(X_train',y_train',h,lambda);
    % p_mlp_clipped = max(0, min(1, p_mlp)); % clip the values between [0,1];
    t_train = toc();

    % Compute MSE Loss;
    tic()
    n_test = size(X_test, 1)
    y_pred = mlpRegPred(model, X_test')';
    t_test = toc() / n_test;
    mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
    eps = y_pred - y_test;
    l = eps' * eps / n_test;
    fprintf("MSE Loss: %.4f\n", l);
    fprintf("Training time: %s\nTest time per data point: %s\n", t_train, t_test);
end

