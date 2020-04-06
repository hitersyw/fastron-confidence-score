%% Parameters;
close all; clear
rng(0);
format shortE;
init;

input_path = base_dir + "log/%s_n%d.mat";
validated_file = base_dir + "pose/poses_combined_validated.csv";

reachability_dataset = 'reachability_score';
self_collision_dataset = 'self_collision_score';
environment_collision_dataset = 'env_collision_score';
n = 2925;

%% Load Original dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n, false);
[X_self_collision, y_self_collision] = load_dvrk3(input_path, self_collision_dataset, n, false);
[X_env_collision, y_env_collision] = load_dvrk3(input_path, environment_collision_dataset, n, false);

%% maximum score from the model-free dataset
y = [y_reach, y_self_collision, y_env_collision, y_reach + y_self_collision + y_env_collision];
max_original = max(y);

%% Load validated dataset
T = readtable(validated_file);
X = T{:, [1,2,4]};
y_reach_optimal = T{:, 6};
y_self_collision_optimal = T{:, 9};
y_env_collision_optimal = T{:, 12};
y_optimal = [y_reach_optimal, y_self_collision_optimal, y_env_collision_optimal,...
    y_reach_optimal + y_self_collision_optimal + y_env_collision_optimal];
max_optimal = max(y_optimal);

%% Maximum scores table;
score_names = {'reachability','self-collision','env-collision','sum'};
max_scores_table = array2table([max_optimal; max_original],'VariableNames', score_names);

%% Accuracy table
y_optimal_pred = T{:,[5, 8, 11]};
y_optimal_pred = [y_optimal_pred, sum(y_optimal_pred, 2)];
delta = y_optimal_pred - y_optimal;
mse = sum(delta.^2)./ size(delta, 1)
l = y_optimal.*log(y_optimal_pred) + (1-y_optimal).*log(1-y_optimal_pred);
l = [l(:, 1:3), zeros(size(l, 1), 1)]; 
nll = -sum(l, 1) / size(l, 1);
accuracy_table = array2table([mse', nll'],'VariableNames',{'mse', 'nll'}, 'RowNames', score_names)
writetable(accuracy_table,'./results/optimal_poses.xls', 'WriteRowNames',true);
writetable(accuracy_table,'./results/optimal_poses.csv', 'WriteRowNames',true);