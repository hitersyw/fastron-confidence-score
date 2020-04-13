% Plot the costmap of a 3D position of 1 PSM arm
close all; clear
rng(0);
init;
format shortE;

% Configurations that are subject to change
arm = "psm1"; 
datetime = "25_03_2020_10";
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";
base_dir = base_dir + "cone/";
input_path = base_dir + "log/" + data_dir;
input_spec = input_path + "%s_n%d.mat";

reachability_dataset = sprintf('reachability_score_%s', arm);
self_collision_dataset = sprintf('self_collision_score_%s', arm);
environment_collision_dataset = sprintf('env_collision_score_%s', arm);
n = 4096; % number of samples to use for fitting. 
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
y_combined = (y_reach_s + y_self_collision_s + y_env_collision_s) / 3; 

%% Interpolate and then find the max among the theta dimension

% Define grid over 3D space; 
x = linspace(xmin(1), xmax(1), 64); % x
y = linspace(xmin(2), xmax(2), 64); % y
z = linspace(xmin(3), xmax(3), 64); % theta
[Xs, Ys, Zs] = meshgrid(x, y, z);
X_2d = Xs(:, :, 1);
Y_2d = Ys(:, :, 1);

%% Interpolate over the space;
f_reach = griddedInterpolant(xg, yg, zg, y_reach_s, 'cubic', 'cubic');
y_reach_g = f_reach(Xs, Ys, Zs);
y_reach_max = max(y_reach_g, [], 3);

f_self_collision = griddedInterpolant(xg, yg, zg, y_self_collision_s, 'cubic', 'cubic');
y_self_collision_g = f_self_collision(Xs, Ys, Zs);
y_self_collision_max = max(y_self_collision_g, [], 3);

f_env_collision = griddedInterpolant(xg, yg, zg, y_env_collision_s, 'cubic', 'cubic');
y_env_collision_g = f_env_collision(Xs, Ys, Zs);
y_env_collision_max = max(y_env_collision_g, [], 3);

y_combined_g = y_reach_g + y_self_collision_g + y_env_collision_g; 
% Normalize between 0 and 1; 
y_combined_max = max(y_combined_g, [], 3) / 3;

%% Plot figures
figure();
levellist = [0.1 0.25 0.5 0.75 0.9];
% cm = interp1([0 0.5 1]', [0.4039 0.6627 0.8118; 0.9686 0.9686 0.9686;0.9373 0.5412 0.3843;], linspace(0,1,256)');
cm = 'parula';

axis square;
imagesc([xmin(1) xmax(1)], [xmin(2) xmax(2)], y_combined_max, [0 1]); hold on;
[cont, cont_h] = contour(X_2d, Y_2d, y_combined_max); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('X'); 
ylabel('Y');

title('Combined Score');
colormap(cm);
hold on;

%% Search for the global optimum in 2D;
stride = 4;
[x, y, v] = searchGlobalOpt2d(X_2d, Y_2d, y_combined_max, stride);