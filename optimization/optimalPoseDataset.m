% This script is used for finding the optimal pose based on the scores 
% from the model-free dataset. 

close all; clear
rng(0);
init;
format shortE;

arm = "psm1"; 
datetime = "16_04_2020_19";

data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";
base_dir = base_dir + "cone/";
input_path = base_dir + "samples/" + data_dir;
input_spec = input_path + "%s_n%d.mat";

output_path = base_dir + "pose/";

reachability_dataset = sprintf('reachability_score_%s', arm);
self_collision_dataset = sprintf('self_collision_score_%s', arm);
environment_collision_dataset = sprintf('env_collision_score_%s', arm);
n = 4096; % number of samples to use for fitting. 
shuffle = true; % whether to shuffle up the dataset;
n_max = 1; % top n pose to show from the dataset. 
tol = 0.001;

use_fastron = true;
output_name = sprintf("pose_%s_n%d_dataset_%s.csv", datetime, n, arm);
result_path = sprintf("./results/optimal_pose_n%d_dataset_%s.mat", n, arm);

%% Load Dataset
[X_reach, y_reach] = loadDvrk2(input_spec, reachability_dataset, n, use_fastron, true);
[X_self_collision, y_self_collision] = loadDvrk2(input_spec, self_collision_dataset, n, use_fastron, true);
[X_env_collision, y_env_collision] = loadDvrk2(input_spec, environment_collision_dataset, n, use_fastron, true);

assert(all(X_reach == X_self_collision, 'all'));
assert(all(X_reach == X_env_collision, 'all'));
X = X_reach;

%% Combined Score
y = y_reach + y_self_collision + y_env_collision; 
figure; clf;
histogram(y, 10,'Normalization','probability');
title(sprintf('%s, n:%d', "Combined", n));
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
assert(all(max(X) <= xmax + tol));
assert(all(min(X) >= xmin - tol));

%% Extract maximum score pose from dataset
combined_raw_score = y_reach + y_self_collision + y_env_collision;
[max_poses, max_scores_dataset] = maxScorePoses(X, [y_reach, y_self_collision, y_env_collision, combined_raw_score], n_max);
z = 0.6599;
X_out = [max_poses(1:2), z, max_poses(3), max_scores_dataset];

%% Write optimal pose from dataset to path; 
save(result_path, 'max_scores_dataset');
path = output_path + output_name;
fprintf("Write to file: %s", path);
writematrix(X_out, path);
