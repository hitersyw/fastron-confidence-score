function [mdl] = trainSVR(X_train, y_train, X_test, y_test, box_constraint, kernel_scale, epsilon)

%     mdl = fitrsvm(X_train, y_train, 'KernelFunction','rbf', ...
%      'OptimizeHyperparameters','auto',...
%      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%      'expected-improvement-plus'));
    
    mdl = fitrsvm(X_train,y_train,'KernelFunction','rbf','KernelScale', kernel_scale, ...
           'BoxConstraint', box_constraint, ...
           'Epsilon', epsilon);
    % 
    y_pred_train = predict(mdl, X_train);
    y_pred_test = predict(mdl, X_test);
    
    % MSE Loss;
    eps_train = y_pred_train - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);

    eps_test = y_pred_test - y_test;
    l_test = eps_test' * eps_test / size(X_test, 1);
    fprintf("MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
end