%% Save the maximum score from the dataset into a file 
close all; clear
rng(0);
init;
format shortE;

input_path = base_dir + "log/%s_n%d.mat";

reachability_dataset = 'reachability_score';
self_collision_dataset = 'self_collision_score';
environment_collision_dataset = 'env_collision_score';
n = 1053; % number of samples to use for fitting. 
shuffle = true; % whether to shuffle up the dataset;
p_test = 0.1; % percentage of samples used for evaluating test error;
n_max = 1; % top n pose to show from the dataset. 

%% Load Dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n, false);
[X_self_collision, y_self_collision] = load_dvrk3(input_path, self_collision_dataset, n, false);
[X_env_collision, y_env_collision] = load_dvrk3(input_path, environment_collision_dataset, n, false);

assert(all(X_reach == X_self_collision, 'all'));
assert(all(X_reach == X_env_collision, 'all'));
X = X_reach;

%% Extract maximum score pose from dataset
combined_raw_score = y_reach + y_self_collision + y_env_collision;
top_n = 20;
[max_poses, max_scores_dataset] = max_score_poses(X, [y_reach, y_self_collision, y_env_collision, combined_raw_score], n_max);

%% Write optimal pose from dataset to path; 
result_path = "./results/optimal_pose_dataset.mat";
save(result_path, 'max_scores_dataset');
