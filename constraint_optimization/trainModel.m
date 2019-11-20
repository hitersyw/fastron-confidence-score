function [mdl] = trainModel(base_path, dataset, n, model)

    score_dict = load(sprintf(base_path, dataset, n));
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

    X_val = X(ceil(n*p_train+1):n, :);
    y_val = y(ceil(n*p_train+1):n);
    
    if model == "svr"
        mdl = fitrsvm(X_train, y_train,'KernelFunction','rbf', 'KernelScale','auto',...
                'Solver','SMO', 'Epsilon', 0.2, ...
                'Standardize',false, 'verbose',0);
            
        % 
        n_val = size(X_val, 1);
        y_pred = predict(mdl, X_val);
        eps = y_pred - y_val;
        l = eps' * eps / n_val;
        fprintf("MSE Loss: %.4f\n", l);
    end
    % Ignore this for now; 
    % else if model == "rvm"
end