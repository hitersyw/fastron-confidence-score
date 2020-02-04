function [PARAMETER] = trainRVM(X_train, y_train, X_test, y_test, g)
    maxIter = 5000;
    OPTIONS = SB2_UserOptions('iterations', maxIter,...
                              'diagnosticLevel', 1,...
                              'monitor', 100);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);
    % Train;
    % Y = X./g;
    K_train = exp(-g * pdist2(X_train,X_train).^2);
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
        SparseBayes('Gaussian', K_train, y_train, OPTIONS, SETTINGS);
    
    % Get model params;
    w_infer	= zeros(size(X_train,2),1);
    nz_ind = PARAMETER.Relevant;
    w_infer(nz_ind)	= PARAMETER.Value;

    % Predict;
    K_test = exp(-g*pdist2(X_test,X_train(nz_ind,:)).^2);
    y_pred_test = K_test*w_infer(nz_ind);
    y_pred_train = K_train(:, nz_ind)*w_infer(nz_ind);
    
    % Calculate loss;
    eps_train = y_pred_train - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);

    eps_test = y_pred_test - y_test;
    l_test = eps_test' * eps_test / size(X_test, 1);
    fprintf("MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
end