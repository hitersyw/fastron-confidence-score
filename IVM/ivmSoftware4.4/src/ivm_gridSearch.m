% IVM_GRIDSEARCH performs gridsearch for the determination of kernel and
% regularization parameter
%
%   Authors:
%     Ribana Roscher(ribana.roscher@fu-berlin.de)
%   Date: 
%     November 2014 (last modified)

function [result, params] = ivm_gridSearch(data, params)

%% N-fold crossvalidation
% training and test data
[indTest, indTrain] = crossval_class_dependend(data.c, params.CV);
         
%% determine best parameters
store = params.maxIter;
params.maxIter = 100;
all_acc = zeros(numel(params.sigmas), numel(params.lambdas));
ind_s = 0;
for s = params.sigmas
    fprintf('sigma = %.2f\n', s)
    ind_s = ind_s + 1;
    ind_l = 0;
    for l = params.lambdas
        ind_l = ind_l + 1;
        % model parameter
        params.lambda = l;
        params.sigma = s;
        accuracies = 0;
        for c = 1:params.CV
            data.phi_train = data.phi(:, indTrain{c});
            data.c_train   = data.c(     indTrain{c});
            data.phi_test  = data.phi(:,  indTest{c});
            data.c_test    = data.c(      indTest{c});

            % model
            model = ivm_learn(data.phi_train, data.c_train, params); 

            % test accuracy
            result = ivm_predict(model, data.phi_test, data.c_test);

            % accuracies
            accuracies = accuracies + result.oa / params.CV;
        end
        fprintf('Cross Validation Accuracy = %.1f %%\n', accuracies * 100)
        all_acc(ind_s, ind_l) = accuracies;
    end
end
params.maxIter = store;

[val idx] = max(all_acc(:));
[sigma_idx lambda_idx] = ind2sub(size(all_acc), idx);
params.sigma = params.sigmas(sigma_idx);
params.lambda = params.lambdas(lambda_idx);

 % model
fprintf('\nCompute final model.\n') 
result = ivm(data, params); 
result.model.CVacc = val; 

% test accuracy
if isfield(data, 'phit') && isfield(data, 'ct')
    result = ivm_predict(result.model, data.phit, data.ct);
end
if isfield(data, 'phit') && ~isfield(data, 'ct')
    result = ivm_predict(result.model, data.phit, 0);
end
if ~isfield(data, 'phit') && ~isfield(data, 'ct')
end
