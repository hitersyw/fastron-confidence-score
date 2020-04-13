% Script for training self-collision, environment-collision and
% reachability models on the DVRK datasets using interpolation. After training the models and
% plotting, save the workspace variables, which can then be loaded for
% optimization. 
close all; clear
rng(0);
init;
format shortE; 

% TODO: move this into a common configuration file; 
arm = "psm1";
base_dir = base_dir + 'cone/';
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";
input_path = base_dir + "log/" + data_dir;

input_spec = input_path + "%s_n%d.mat";
reachability_dataset = "reachability_score" + "_" + arm;
self_collision_dataset = "self_collision_score" + "_" + arm;
env_collision_dataset = "env_collision_score" + "_" + arm;
use_fastron = true;
n = 64;
n_test = 4096;
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 
tol = 0.001;
tune_params = false;
result_path = sprintf("./results/interpolation_training_n%d_%s.mat", n, arm);

%% Load Training Dataset
[X_reach_train, y_reach_train] = loadDvrk2(input_spec, reachability_dataset, n, use_fastron,false);
[X_self_collision_train, y_self_collision_train] = loadDvrk2(input_spec, self_collision_dataset, n, use_fastron, false);
[X_env_collision_train, y_env_collision_train] = loadDvrk2(input_spec, env_collision_dataset, n, use_fastron, false);

% safety check;
assert(all(X_reach_train == X_self_collision_train, 'all'));
assert(all(X_reach_train == X_env_collision_train, 'all'));
X_train = X_reach_train;

%% Load workspace limit and safety check
run(input_path + "workspace_limit.m");
% xmax = max(X_train);
% xmin = min(X_train);

if arm == "psm1"
    xmax = xmax_psm1;
    xmin = xmin_psm1;
elseif arm == "psm2"
    xmax = xmax_psm2;
    xmin = xmin_psm2;
end
assert(all(max(X_train) <= xmax + tol));
assert(all(min(X_train) >= xmin - tol));

%% Load Test Dataset and safety check;
[X_reach_test, y_reach_test] = loadDvrk2(input_spec, reachability_dataset, n_test, use_fastron,false);
[X_self_collision_test, y_self_collision_test] = loadDvrk2(input_spec, self_collision_dataset, n_test, use_fastron,false);
[X_env_collision_test, y_env_collision_test] =  loadDvrk2(input_spec, env_collision_dataset, n_test, use_fastron,false);
% Safety Check;
assert(all(X_reach_test == X_self_collision_test, 'all'));
assert(all(X_reach_test == X_env_collision_test, 'all'));
assert(all(max(X_reach_test) <= xmax + tol));
assert(all(min(X_reach_test) >= xmin - tol));

X_test = X_reach_test;

%% Extract maximum score poses from dataset
combined_raw_score = y_reach_train + y_self_collision_train + y_env_collision_train;
top_n = 10;
[max_poses, max_scores] = maxScorePoses(X_train, [y_reach_train, y_self_collision_train, y_env_collision_train, combined_raw_score], n_max);
display("Displaying maximum poses and their scores");
display([max_poses, max_scores]);

%% Normalize the dataset;
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Build interpolation models;
tic();

xs = numel(unique(X_train(:, 1)));
ys = numel(unique(X_train(:, 2)));
zs = numel(unique(X_train(:, 3)));

% Permuation from Python
xg = permute(reshape(X_train(:, 1), [zs, ys, xs]), [3 2 1]);
yg = permute(reshape(X_train(:, 2), [zs, ys, xs]), [3 2 1]);
zg = permute(reshape(X_train(:, 3), [zs, ys, xs]), [3 2 1]);

% % Permutation from Matlab
% xg = permute(reshape(X_train(:, 1), [zs, ys, xs]), [2, 1, 3]);
% yg = permute(reshape(X_train(:, 2), [zs, ys, xs]), [2, 1, 3]);
% zg = permute(reshape(X_train(:, 3), [zs, ys, xs]), [2, 1, 3]);

%%
y_reach_train_s = permute(reshape(y_reach_train, [zs, ys, xs]), [3 2 1]); % TODO: Use grid config instead of this hardcode;
y_self_collision_train_s = permute(reshape(y_self_collision_train, [zs, ys, xs]), [3 2 1]);
y_env_collision_train_s = permute(reshape(y_env_collision_train, [zs, ys, xs]), [3 2 1]);

F_reach = griddedInterpolant(xg, yg, zg, y_reach_train_s, 'cubic', 'cubic');
t_reach_train = toc();

tic();
F_self_collision = griddedInterpolant(xg, yg, zg, y_self_collision_train_s, 'cubic', 'cubic');
t_self_collision_train = toc();

tic();
F_env_collision = griddedInterpolant(xg, yg, zg, y_env_collision_train_s, 'cubic', 'cubic');
t_env_collision_train = toc();

%% Test model performance
tic();
y_reach_pred = F_reach(X_test);
t_reach_test = toc() / size(X_test, 1);

tic();
y_self_collision_pred = F_self_collision(X_test);
t_self_collision_test = toc() / size(X_test, 1); 

tic();
y_env_collision_pred = F_env_collision(X_test);
t_env_collision_test = toc() / size(X_test, 1);

eps_reach = y_reach_pred - y_reach_test;
eps_self_collision = y_self_collision_pred -y_self_collision_test;
eps_env_collision = y_env_collision_pred - y_env_collision_test;

mse_reach = eps_reach'*eps_reach/n_test;
mse_self_collision =  eps_self_collision'*eps_self_collision/n_test;
mse_env_collision = eps_env_collision'*eps_env_collision/n_test;

fprintf("MSE loss for reachability: %.4f; Maximum error: %.4f\n", mse_reach, max(abs(eps_reach)));
fprintf("MSE loss for self-collision: %.4f; Maximum error: %.4f\n", mse_self_collision, max(abs(eps_self_collision)));
fprintf("MSE loss for env collision: %.4f; Maximum error: %.4f\n", mse_env_collision, max(abs(eps_env_collision)));

%% Write out performance for plotting;

% [mse; max_absolute_difference; training time; test me per sample];
T = [mse_reach, mse_self_collision, mse_env_collision; ...
     max(abs(eps_reach)), max(abs(eps_self_collision)), max(abs(eps_env_collision)); ...
     t_reach_train, t_self_collision_train, t_env_collision_train; ...
     t_reach_test, t_self_collision_test, t_env_collision_test];
save(result_path, 'T');