% Training procedure for platt scaling;
% F: Hypothesis functions of the validation set;
% y: Labels for the validation set;
% A, B: previously trained parameters;
% lr: Learning rate; 
function [A, B] = trainPlattScaling(F, y, maxIter, eps, lr, A, B)
    % Train A, B from scratch;
    if nargin < 6
        A = 0; B = 0;
    end
    
    p = 1./(1 + exp(A * F + B));
    p = clip(p, 0.001);
    t = (y + 1) / 2;
    t(t==1) = (sum(t==1) + 1)/(sum(t==1) + 2);
    t(t==0) = 1./(sum(t==0)+2); % see paper for treatment of the target values;
    l = -(t'*log(p)+(1-t)'*log(1-p));
    display(sprintf('Initial Loss: %f',l));
    
    i = 0;
    old_l = realmax;
    errors_d = [old_l - l];
    start = 1;
    %while i < maxIter & old_l - l > eps
    while i < maxIter & sum(errors_d(start:end)<eps)< 10
        old_l = l;
        % gradient descent rule;
        A = A + lr * F' * (p - t); 
        B = B + lr * sum(p - t); 
        p = 1./(1 + exp(A * F + B));
        l = -(t'*log(p)+(1-t)'*log(1-p));
        % display(sprintf('Loss at the %d iteration: %f', i, l));
        i = i + 1;
        errors_d = [errors_d, old_l - l]; 
        start = max(1,size(errors_d,2)-9);
    end
    
    if i >= maxIter
        display("Max Iteration has reached!");
    else
        display(sprintf("Error has not improved over %f for 10 times after %d iteration", eps, i));
    end       
end