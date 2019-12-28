function [mdl] = svrTrain(X_train,y_train, X_test, y_test)    
    mdl = fitrsvm(X_train, y_train,'KernelFunction','rbf', 'KernelScale',0.1,...
            'Solver','SMO', 'Epsilon', 0.02, 'BoxConstraint', 5, ...
            'Standardize',false, 'verbose',0);

    % 
    eps_train = predict(mdl, X_train) - y_train;
    l_train = eps_train' * eps_train / size(X_train, 1);

    eps_test = predict(mdl, X_test) - y_test;
    l_test = eps_test' * eps_test / size(X_test, 1);
    fprintf("SVR MSE Loss: %.4f (training); %.4f (test)\n", l_train, l_test);
end