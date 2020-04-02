function [mdl] = svrTrain2(X_train,y_train, X_test, y_test)
    mdl = fitrsvm(X_train,y_train,'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5);
    
    % scale_output = @(y) 1/max(y)*max(y,0); 
    scale_output = @(y) y;
    y_pred_train = predict(mdl, X_train);
    y_pred_test = predict(mdl, X_test);
    
    % Compute losses; 
    eps_train = scale_output(y_pred_train) - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);

    eps_test = scale_output(y_pred_test) - y_test;
    l_test = eps_test' * eps_test / size(X_test, 1);
    fprintf("SVR MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
end