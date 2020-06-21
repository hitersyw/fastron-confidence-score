% Plot the costmap of a 3D position of 1 PSM arm
close all; clear
rng(0);
init;
format shortE;
set(0,'defaultTextInterpreter','latex'); 

% Configurations that are subject to change
arm = "psm1"; 
datetime = "23_04_2020_11";
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik_2/";
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

xg = permute(reshape(X(:, 1), [zs, ys, xs]), [3 2 1]);
yg = permute(reshape(X(:, 2), [zs, ys, xs]), [3 2 1]);
zg = permute(reshape(X(:, 3), [zs, ys, xs]), [3 2 1]);

y_reach_s = permute(reshape(y_reach, [zs, ys, xs]), [3 2 1]); % TODO: Use grid config instead of this hardcode;
y_self_collision_s = permute(reshape(y_self_collision, [zs, ys, xs]), [3 2 1]);
y_env_collision_s = permute(reshape(y_env_collision, [zs, ys, xs]), [3 2 1]);
y_combined = y_reach_s + y_self_collision_s + y_env_collision_s; 

y_reach_max = max(y_reach_s, [], 3);
y_self_collision_max = max(y_self_collision_s, [], 3);
y_env_collision_max = max(y_env_collision_s, [], 3);
y_combined_max = max(y_combined, [], 3) / 3;

%% Plot figures
nx = 200; ny = 200;
[X_2d, Y_2d] = meshgrid(linspace(xmin(1), xmax(1), nx), linspace(xmin(2), xmax(2), ny));

%% Reachability
figure();
levellist = [0.1 0.25 0.5 0.75 0.9];
% cm = interp1([0 0.5 1]', [0.4039 0.6627 0.8118; 0.9686 0.9686 0.9686;0.9373 0.5412 0.3843;], linspace(0,1,256)');
cm = 'parula';
axis square;
img_reach = imresize(y_reach_max, [nx, ny])';
imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], img_reach, [0 1]); hold on;
% img = imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], y_reach_max, [0 1]); hold on;
% Draw contours;
[cont, cont_h] = contour(X_2d, Y_2d, img_reach); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('X'); 
ylabel('Y');  
title('Reachability');
hold on;
colormap(cm);

%% Self-Collision;
figure();
axis square;

img_self_collision = imresize(y_self_collision_max, [nx, ny])';
imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], img_self_collision, [0 1]); hold on;
[cont, cont_h] = contour(X_2d, Y_2d, img_self_collision); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('X'); 
ylabel('Y');

title('Self Collision');
hold on;
colormap(cm);

% Env collision;
figure();
axis square;
img_env_collision = imresize(y_env_collision_max, [nx, ny])';
imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], img_env_collision, [0 1]); hold on;
[cont, cont_h] = contour(X_2d, Y_2d, img_env_collision); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('X'); 
ylabel('Y');

title('Env Collision');
hold on;
colormap(cm);

% Combined Score
figure();
axis square;
img_combined = imresize(y_combined_max, [nx, ny])';
imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], img_combined, [0 1]); hold on;
[cont, cont_h] = contour(X_2d, Y_2d, img_combined); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('X'); 
ylabel('Y');
title('Combined Score');
hold on;
colormap(cm); hold on;

%% Load the optimal solution found by each method; 
n_sparse = 64; n_dense = 4096;
date = "16_04_2020_19";
file_spec = "./dvrkData/cone/pose/pose_%s_n%d_%s_%s_validated.mat";
methods = ["weightedSVR", "interpolation", "dataset", "coarse2fine", "dataset"];
nums = [n_sparse, n_sparse, n_sparse, n_dense, n_dense];

pos = zeros(numel(methods), 3);
for i = 1:numel(methods)
    load(sprintf(file_spec, date, nums(i), methods(i), arm));
    T = validated_score(1:3);
    pos(i, :) = T;
end

sz = [20, 20, 20, 50, 20]; % Might have overlapping positions
colors = [0, 0, 1; 0, 1, 0; 1, 0, 1; 1, 0, 0; 0, 0, 0];
scArray = [];
for i=1:numel(methods)
    sc = scatter(pos(i, 1), pos(i, 2), sz(i), colors(i,:), 'filled'); hold on;
    scArray = [scArray, sc];
end

labels = methods;
labels(3) = "sparse"; labels(5) = "dense";
legend(scArray, cellstr(labels));