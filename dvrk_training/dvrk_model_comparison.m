clear; close all;
rng(0);
format shortE;
init;

input_path = base_dir + "log/%s_n%d.mat";

% define parameters 
models = ["SVR", "RVM", "MLP"];
training_datasets = ["reachability_score", "self_collision_score", "env_collision_score"];
normalized = true;
g = 1;
% test_dataset = 'reachability_score_test'; 
n = 2925;

metrics = zeros(numel(training_datasets) * numel(models), 4);
row_names = [];

% additional time for pre-process and post-process; 
t_pre_process = 0;
t_post_process = 0;
for i = 1:numel(training_datasets)
    training_dataset = training_datasets(i);
    
    % Load dataset
    [X, y] = load_dvrk3(input_path, training_dataset, n, false);
    
    % Normalize datasets
    if normalized
        xmax = max(X);
        xmin = min(X);
        % scale_input = @(x) x;
        
        tic();
        scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
        X = scale_input(X);
        t_pre_process = toc();
    end
    
    % Iterate over models
    for j = 1:numel(models)
        model = models(j);
        
        fprintf("Training %s on %s dataset", model, training_dataset);
        % Train model
        if model == "SVR"
            % Train;
            tic();
            mdl = fitrsvm(X, y, 'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5, 'IterationLimit', 5000);
            training_time = toc();

            % Predict;
            tic();
            y_pred = predict(mdl, X);
            test_time = toc();
        elseif model == "RVM"
            % model parameters
            maxIter = 5000;
            OPTIONS = SB2_UserOptions('iterations', maxIter,...
                                      'diagnosticLevel', 1,...
                                      'monitor', 100);
            SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);
            % Train;
            tic();
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
            test_time = toc();
            
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
            test_time = toc();
        else
            fprintf("Model not supported: %s", mdl);
            return
        end
        
        % Scale output
        ymin = min(y);
        ymax = max(y);
        
        tic();
        if ymax - ymin ~= 0
            y_pred = (y_pred - ymin)./(ymax - ymin);
        end
        y_pred = clip(y_pred, 0.0001);
        t_post_process = toc(); 

        % Add pre-process and post-process time to training & test time;
        training_time = t_pre_process + training_time + t_post_process;
        test_time = (t_pre_process + test_time + t_post_process) / size(X, 1);
        fprintf("Training time %s\n", training_time);
        fprintf("Test time per sample %s\n", test_time);

        % Compute loss;
        % MSE Loss;
        eps = y_pred - y;
        mse = eps' * eps / size(X, 1);
        fprintf("MSE Loss for %s: %s\n", model, mse);

        % NLL loss;
        nll_s = y.*log(y_pred)+(1-y).*log(1-y_pred);
        nll = -sum(nll_s)/size(y,1);
        fprintf("NLL Loss for %s: %s\n", model, nll);
        
        index = (i-1)* numel(models) + j;
        metrics(index, 1) = mse;
        metrics(index, 2) = nll;
        metrics(index, 3) = training_time;
        metrics(index, 4) = test_time;
        
        ds = strrep(training_dataset, '_', '-');
        row_names = [row_names, model + ' - ' +  training_dataset];  
    end
end


%% Create table; 
col_names = {'MSE', 'NLL', 'Training Time (sec)', 'Test time (sec/sample)'};
T1 = array2table(metrics, ...
    'VariableNames', col_names, ...
    'RowNames', row_names);
writetable(T1,'./results/regression_models.xls', 'WriteRowNames',true)
writetable(T1,'./results/regression_models.csv', 'WriteRowNames',true)

%% Create table for model-based and model-free comparison
% Use only SVR for comparison with model-free method; 
num_checks = 88.65;
row_indices = numel(models).* linspace(0, numel(training_datasets) - 1, numel(training_datasets)) ...
    + find(models == "SVR");
col_index = find(col_names == "Test time (sec/sample)");
row_names_T2 = strrep(row_names(row_indices), 'SVR', 'Model-based');
row_names_T2 = strrep(row_names_T2, '_score', '');
row_names_T2 = strrep(row_names_T2, '_', '-');
total_checks = metrics(row_indices, col_index) * 88.65;
T2 = array2table([metrics(row_indices, col_index), total_checks], ...
    'VariableNames', {'Time per Check (sec)', 'Total Time (sec'}, ...
    'RowNames', row_names_T2);

writetable(T2,'./results/model_based.xls', 'WriteRowNames',true)
