clear; close all;
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";

% define parameters 
model = "MLP";
normalized = true;
training_dataset = 'self_collision_score';
g = 1;
% test_dataset = 'reachability_score_test'; 
n = 2925;

%% Load Dataset
[X, y] = load_dvrk3(input_path, training_dataset, n, false);

%% Normalize Datasets
if normalized
    xmax = max(X);
    xmin = min(X);
    % scale_input = @(x) x;
    scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
    X = scale_input(X);
end

%% Train model
if model == "SVR"
    % Train;
    tic();
    mdl = fitrsvm(X, y, 'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5, 'IterationLimit', 5000);
    training_time = toc();
    
    % Predict;
    tic();
    y_pred = predict(mdl, X);
    test_time = toc() / size(X, 1);
elseif model == "RVM"
    % model parameters
    maxIter = 5000;
    OPTIONS = SB2_UserOptions('iterations', maxIter,...
                              'diagnosticLevel', 2,...
                              'monitor', 100);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);
    % Train;
    tic()
    % Y = X./g;
    K = exp(-g * pdist2(X,X).^2);
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
        SparseBayes('Gaussian', K, y, OPTIONS, SETTINGS);
    training_time = toc();
    
    % Get model params;
    w_infer	= zeros(size(X,2),1);
    nz_ind = PARAMETER.Relevant;
    w_infer(nz_ind)	= PARAMETER.Value;
    
    % Predict;
    tic();
    K_test = exp(-g*pdist2(X,X(nz_ind,:)).^2);
    y_pred = K_test*w_infer(nz_ind);
    test_time = toc() / size(X, 1);
elseif model == "MLP"
    % define model parameter
    h = [128,128];            % two hidden layers with 64 and 64 neurons
    lambda = 0.0001;        % Regularized Loss; 
    
    % train
    tic();
    [mdl, L] = mlpRegSigmoid(X',y',h,lambda);
    training_time = toc();
    
    % test
    tic();
    y_pred = mlpRegPredSigmoid(mdl, X')';
    test_time = toc() / size(X, 1);
else
    fprintf("Model not supported: %s", mdl);
    return
end

% Scale output
ymin = min(y);
ymax = max(y);
if ymax - ymin ~= 0
    y_pred = (y_pred - ymin)./(ymax - ymin);
end
y_pred = clip(y_pred, 0.0001);

fprintf("Training time %.4f\n", training_time);
fprintf("Test time per sample %.10f\n", test_time);

%% Compute loss;
% MSE Loss;
eps = y_pred - y;
mse = eps' * eps / size(X, 1);
fprintf("MSE Loss for %s: %.8f\n", model, mse);

% NLL loss;
nll_s = y.*log(y_pred)+(1-y).*log(1-y_pred);
nll = -sum(nll_s)/size(y,1);
fprintf("NLL Loss for %s: %.8f\n", model, nll);