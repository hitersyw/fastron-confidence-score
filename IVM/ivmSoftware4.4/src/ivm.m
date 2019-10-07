% IVM train and test the IVM classifier
%
%   IVM trains the IVM model and classify test data, whereas the output is
%   stored in the struct RESULT:
%       RESULT.P: probabilities of test data
%       RESULT.trainTime: training time
%       RESULT.testTime: testing time
%       RESULT.confMat: confusion matrix (rows: true label, columns: estimated label)
%       RESULT.perc: User and Producer accuracy
%       RESULT.oa: overall accuracy
%       RESULT.aa: average accuracy
%       RESULT.aa_c: class-wise average accuracy
%       RESULT.kappa: kappa coefficient
%       RESULT.nIV: number of used import vectors
%       RESULT.Ntest: number of testing data
%       RESULT.model: stored model
%           RESULT.model.indTrain: indices of used training points
%           RESULT.model.P: probabilities of train data
%           RESULT.model.lambda: regularization parameter
%           RESULT.model.trainError: absolute training error
%           RESULT.model.params: used params (output from init)
%           RESULT.model.trainTime: training time
%           RESULT.model.IV: feature vectors of the import vectors
%           RESULT.model.kernelSigma: used kernel parameter
%           RESULT.model.lambda: used regularization parameter
%           RESULT.model.C: number of classes
%           RESULT.model.c: true labels of import vectors
%           RESULT.model.S: indices of the import vectors
%           RESULT.model.nIV: number of used import vectors
%           RESULT.model.alpha: parameters of the decision hyperplane
%           RESULT.model.fval: last function value of objective function
%           RESULT.model.CVacc: highest crossvalidation value
%   The input is the struct DATA with DATA.PHI being the (MxN)-dimensional 
%   features with M being the feature dimensionand N being the number of 
%   samples, and DATA.C being a (Nx1)-dimensional label vector C filled 
%   with labels 1,...,C with C being the number of classes; optional are 
%   DATA.PHIT being the (MxT)-dimensional testing data with M dimensions 
%   and T training samples, and DATA.CT the test labels; PARAMS is a struct
%   initialized in init.m containing the relevant hyperparameters
%   
%   Authors:
%     Ribana Roscher(ribana.roscher@fu-berlin.de)
%   Date: 
%     November 2014 (last modified)

%%
function result = ivm(data, params)

%% check data struct
perform_predict = 1;
% check data struct
if ~isfield(data, 'phi')
    error('training data is not given')
end
if ~isfield(data, 'c')
    error('training labels are not given')
end
% if no test dat is given
if ~isfield(data, 'phit')
    disp('info: test labels and/or test data are not given.')
    perform_predict = 0;
end
% if only no test labels are given
if isfield(data, 'phit') && ~isfield(data, 'ct')
    disp('info: test labels are not given. Cannot evaluate test data.')
    perform_predict = 2;
end

% check dimensions
N = size(data.phi, 2);
if size(data.c, 1) ~= N
    data.c = data.c';
end
if numel(data.c) ~= N
    error('dimension of train feature matrix and train label vector does not fit')
end

if perform_predict == 1
	if size(data.ct, 1) ~= size(data.phit, 2)
        data.ct = data.ct';
    end
    if numel(data.ct) ~= size(data.phit, 2)
        error('dimension of test feature matrix and test label vector does not fit')
    end
end

%% learn model and predict probabilties of test data, else 
%% N-fold crossvalidation if parameters are not given
if isfield(params, 'lambda') && isfield(params, 'sigma')
    
    % model
    model = ivm_learn(data.phi, data.c, params); 
    fprintf('Train Accuracy = %.1f %% (%.0f/%.0f)\n', ...
        100 - model.trainError / numel(data.c) * 100,...
        numel(data.c) - model.trainError, numel(data.c))

    % test accuracy
    result.model = model;
    if perform_predict == 1
        result = ivm_predict(result.model, data.phit, data.ct);
        fprintf('Test Accuracy = %.1f %% (%.0f/%.0f)\n', ...
            result.oa * 100, sum(diag(result.confMat)), result.Ntest)
    end
    if perform_predict == 2
        result = ivm_predict(result.model, data.phit, 0);
    end
else % without parameter
    storeNadd = params.Nadd;
    params.Nadd = 50;
    [result, params] = ivm_gridSearch(data, params);
    
    fprintf('\n optimal parameters:\n kernel parameter sigma = %.2f\n regularization parameter lambda: exp(%s)\n\n', ...
            result.model.kernelSigma, num2str(log(result.model.lambda)))
    params.Nadd = storeNadd;    
end
