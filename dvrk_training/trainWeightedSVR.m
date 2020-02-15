function [mdl] = trainWeightedSVR(X_train, y_train, X_test, y_test, observation_weights, box_constraint, kernel_scale, epsilon)

%     mdl = fitrsvm(X_train, y_train, 'KernelFunction','rbf', ...
%      'OptimizeHyperparameters','auto',...
%      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%      'expected-improvement-plus'));
    
    mdl = fitrsvm(X_train,y_train,'KernelFunction','rbf','KernelScale', kernel_scale, ...
           'BoxConstraint', box_constraint, ...
           'Epsilon', epsilon, ...
           'Weights', observation_weights ...
           );
    % 
    y_pred_train = predict(mdl, X_train);
    y_pred_test = predict(mdl, X_test);
    
    % MSE Loss;
    eps_train = y_pred_train - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);
    
    [~, max_train_ind] = max(y_train);
    [~, max_test_ind] = max(y_test); 

    eps_test = y_pred_test - y_test;
    l_test = eps_test' * eps_test / size(X_test, 1);
    fprintf("MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
    fprintf("Maximum loss: %.4f (training); %.4f (test)\n", max(abs(eps_train)), max(abs(eps_test)));
    fprintf("Loss at the maximum score: %.4f (training); %.4f (test)\n", abs(eps_train(max_train_ind)), abs(eps_test(max_test_ind))); 
end