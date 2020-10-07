% Test for training SVR on the dVRK datasets.
%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "samples/workspace_x0.25_0.15_y0.15_0.25/%s_n%d.mat";
n_total = 640;
s = 1;
scale_label = @(y) s.*y;
scale_inverse = @(y) y./s;
tol = 0.075;
% n_original = 640;
% result_path = sprintf('./results/svr_training_%d.mat', n_total);

% training_set = sprintf('reachability_score_%d', n_total);
dataset = 'self_collision_score';
n_test = 1053;
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 

%% Load workspace limit
run("./dvrk_data/workspace_limit.m");

%% Load Dataset
[X_train, y_train] = load_dvrk3(input_path, dataset, n_total, false);
n = size(X_train, 1);

[X_test, y_test] = load_dvrk3(input_path, dataset, n_test, false);

% Safety check
assert(all(max(X_train) <= xmax + tol));
assert(all(min(X_train) >= xmin - tol));
%% Normalize the dataset;
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;

X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Shuffle and divide up the dataset;
tic();
weights = quadratic(1.0,0.5,0.5,y_train);
fprintf("Maximum weight ratio: %s", max(weights) / min(weights));
% reachability
mdl = trainWeightedSVR(X_train, scale_label(y_train), X_test, scale_label(y_test), true, weights, 1.3106, 0.97917, 0.0038211);
% collision;
% collision_mdl = trainWeightedSVR(X_train, y_train, X_test, y_test, true, weights, 869.79, 0.49461, 0.044427);
% env_collision
% env_collision_mdl = trainWeightedSVR(X_train, y_train, X_test, y_test, true, weights, 2.268, 0.63832, 9.5223e-05);
t_train = toc();

%% Compute test loss
tic();
y_pred = scale_inverse(predict(mdl, X_test));
l = y_pred - y_test;
mse = l' * l / size(X_test, 1);
t_test = toc() / size(X_test, 1);
abs_loss_grid = abs(l);

%% Plot error of model fitting
figure('Position', [327 87 800 800]);
ind = abs_loss_grid > 0.1;

% reachability;
cm = jet(256);
subplot(1,2,1)
colorScatter3(X_test(ind,1),X_test(ind,2),...
    X_test(ind,3),abs_loss_grid(ind), cm);
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
colorScatter3(X_test(:,1),X_test(:,2),...
    X_test(:,3),y_pred, cm);
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