function [mdl] = trainModel(X_train, y_train, X_test, y_test, model)
    if model == "svr"
%         mdl = fitrsvm(X_train, y_train,'KernelFunction','rbf', 'KernelScale','auto',...
%                 'Solver','SMO', 'Epsilon', 0.05, ...
%                 'Standardize',false, 'verbose',0);
        
        mdl = fitrsvm(X_train,y_train,'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5);
        % 
        y_pred_train = predict(mdl, X_train);
        y_pred_test = predict(mdl, X_test);
        eps_train = y_pred_train - y_train;
        l_train = eps_train' * eps_train / size(X_train, 1);
        
        eps_test = y_pred_test - y_test;
        l_test = eps_test' * eps_test / size(X_test, 1);
        fprintf("MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
    end
    % Ignore this for now; 
    % else if model == "rvm"
end