% This script is used for searching the global optimum of the 
% 3-dim Dvrk datasets, using coarse-to-fine sampling selection.
close all; clear
rng(0);
init;
format shortE;
 
% Configurations that are subject to change
arm = "psm2"; 
datetime = "25_03_2020_10";
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";
base_dir = base_dir + "cone/";
input_path = base_dir + "log/" + data_dir;
input_spec = input_path + "%s_n%d.mat";

reachability_dataset = sprintf('reachability_score_%s', arm);
self_collision_dataset = sprintf('self_collision_score_%s', arm);
environment_collision_dataset = sprintf('env_collision_score_%s', arm);
n = 64; % number of samples to use for fitting. 
tol = 0.001;
use_fastron = true;

%% Load Dataset
[X_reach, y_reach] = loadDvrk2(input_spec, reachability_dataset, n, use_fastron, false);
[X_self_collision, y_self_collision] = loadDvrk2(input_spec, self_collision_dataset, n, use_fastron, false);
[X_env_collision, y_env_collision] = loadDvrk2(input_spec, environment_collision_dataset, n, use_fastron, false);

assert(all(X_reach == X_self_collision, 'all'));
assert(all(X_reach == X_env_collision, 'all'));
X = X_reach;

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

%% Convert column-based format into a grid
xs = numel(unique(X(:, 1)));
ys = numel(unique(X(:, 2)));
zs = numel(unique(X(:, 3)));

% Permutation from Matlab
xg = permute(reshape(X(:, 1), [zs, ys, xs]), [3 2 1]);
yg = permute(reshape(X(:, 2), [zs, ys, xs]), [3 2 1]);
zg = permute(reshape(X(:, 3), [zs, ys, xs]), [3 2 1]);

% TODO: Use grid config instead of this hardcode;
y_reach_s = permute(reshape(y_reach, [zs, ys, xs]), [3 2 1]); 
y_self_collision_s = permute(reshape(y_self_collision, [zs, ys, xs]), [3 2 1]);
y_env_collision_s = permute(reshape(y_env_collision, [zs, ys, xs]), [3 2 1]);
y_combined = y_reach_s + y_self_collision_s + y_env_collision_s; 

%% Search for the global optimum in 3D;
stride = 1; 
[x, y, z, v] = searchGlobalOpt3d(xg, yg, zg, y_combined, stride);