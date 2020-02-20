%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";
n_total = 819;
n_original = 640;
result_path = sprintf('./results/svr_training_smote%d_%d.mat', n_total, n_original);

reachability_dataset = sprintf('reachability_score_smote%d_3', n_total);
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 
p_test = 0.1;

%% Load Dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n_original, false);
X = X_reach;
n = size(X, 1);

%% Normalize the dataset;
xmax = max(X);
xmin = min(X);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X = scale_input(X);

%% Shuffle and divide up the dataset;
if shuffle
    % shuffle the dataset;
    idx = randperm(n); 
    X = X(idx, :);
    y_reach = y_reach(idx);
end

% Test set; 
X_test = X(1:ceil(n*p_test), :);
y_reach_test = y_reach(1:ceil(n*p_test));

% Training set;
X_train = X(ceil(n*p_test+1):n, :);
y_reach_train = y_reach(ceil(n*p_test+1):n);

tic();
weights = quadratic(1, 0.5, y_reach_train);
reachability_mdl = trainWeightedSVR(X_train, y_reach_train, X_test, y_reach_test, true, weights, 1.589, 0.33453, 0.00054715);
t_reach_train = toc();

%% Compute test loss
tic();
y_pred = predict(reachability_mdl, X_test);
l_reach = y_pred - y_reach_test;
mse_reach = l_reach' * l_reach / size(X_test, 1);
t_reach_test = toc() / size(X_test, 1);

y_pred_grid = predict(reachability_mdl, X);
abs_loss_grid = abs(y_pred_grid - y_reach);


%% Plot error of model fitting
figure('Position', [327 87 800 800]);
ind = abs_loss_grid > 0.05;

% reachability;
cm = jet(256);
subplot(1,2,1)
colorScatter3(X(ind,1),X(ind,2),...
    X(ind,3),abs_loss_grid(ind), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Error plot");
    
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');

subplot(1,2,2)
colorScatter3(X(:,1),X(:,2),...
    X(:,3),y_pred_grid, cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Reachability plot");
    
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Prediction plot");
sgtitle("Model-based scores grid");

%% Save model
T = [mse_reach, max(abs(l_reach)), t_reach_train, t_reach_test];
save(result_path, 'T');