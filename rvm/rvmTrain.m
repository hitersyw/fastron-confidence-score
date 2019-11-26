function [w_infer] = rvmTrain(K_train, y_train, K_test, y_test)
    M = size(K_train,2);

    %% Define model parameters
    maxIter = 500;
    OPTIONS = SB2_UserOptions('iterations', maxIter,...
                              'diagnosticLevel', 2,...
                              'monitor', 10);
    SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
        SparseBayes('Gaussian', K_train, y_train, OPTIONS, SETTINGS);
    w_infer						= zeros(M,1);
    w_infer(PARAMETER.Relevant)	= PARAMETER.Value;
    
    %% Compute training loss;
    y_pred_train = K_train*w_infer;
    eps_train = y_pred_train - y_train;
    l_train = eps_train' * eps_train / size(y_train, 1);

    %% Compute test loss;
    y_pred_test = K_test*w_infer;
    eps_test = y_pred_test - y_test;
    l_test = eps_test' * eps_test / size(y_test, 1);
    fprintf("RVM MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
end