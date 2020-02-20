%% Parameters;
close all; clear
rng(0);
init;
format shortE; 

input_path = base_dir + "log/%s_n%d.mat";

reachability_dataset = 'reachability_score';
self_collision_dataset = 'self_collision_score';
environment_collision_dataset = 'env_collision_score';
n = 640; % number of samples to use for fitting. 
shuffle = true; % whether to shuffle up the dataset;
p_test = 0.1; % percentage of samples used for evaluating test error;
n_max = 10; % top n pose to show from the dataset. 

%% Load Dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n, false);
[X_self_collision, y_self_collision] = load_dvrk3(input_path, self_collision_dataset, n, false);
[X_env_collision, y_env_collision] = load_dvrk3(input_path, environment_collision_dataset, n, false);

assert(all(X_reach == X_self_collision, 'all'));
assert(all(X_reach == X_env_collision, 'all'));
X = X_reach;

%% Extract maximum score poses from dataset
combined_raw_score = y_reach + y_self_collision + y_env_collision;
top_n = 20;
[max_poses, max_scores] = max_score_poses(X, [y_reach, y_self_collision, y_env_collision, combined_raw_score], n_max);
display("Displaying maximum poses and their scores");
display([max_poses, max_scores]);

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
    y_self_collision = y_self_collision(idx);
    y_env_collision = y_env_collision(idx);
    y_reach = y_reach(idx);
end

% Test set; 
X_test = X(1:ceil(n*p_test), :);
y_reach_test = y_reach(1:ceil(n*p_test));
y_self_collision_test = y_self_collision(1:ceil(n*p_test));
y_env_collision_test = y_env_collision(1:ceil(n*p_test));

% Training set;
X_train = X(ceil(n*p_test+1):n, :);
y_reach_train = y_reach(ceil(n*p_test+1):n);
y_self_collision_train = y_self_collision(ceil(n*p_test+1):n);
y_env_collision_train = y_env_collision(ceil(n*p_test+1):n);

%% Build interpolation models;
tic();
F_reach = scatteredInterpolant(X_train, y_reach_train, 'linear', 'linear');
t_reach_train = toc();

tic();
F_self_collision = scatteredInterpolant(X_train, y_self_collision_train, 'linear', 'linear');
t_self_collision_train = toc();

tic();
F_env_collision = scatteredInterpolant(X_train, y_env_collision_train, 'linear', 'linear');
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
n = numel(y_reach_test);

mse_reach = eps_reach'*eps_reach/n;
mse_self_collision =  eps_self_collision'*eps_self_collision/n;
mse_env_collision = eps_env_collision'*eps_env_collision/n;

fprintf("MSE loss for reachability: %.4f; Maximum error: %.4f\n", mse_reach, max(abs(eps_reach)));
fprintf("MSE loss for self-collision: %.4f; Maximum error: %.4f\n", mse_self_collision, max(abs(eps_self_collision)));
fprintf("MSE loss for env collision: %.4f; Maximum error: %.4f\n", mse_env_collision, max(abs(eps_env_collision)));

%% Write out performance for plotting;

% [mse; max_absolute_difference; training time; test time per sample];
T = [mse_reach, mse_self_collision, mse_env_collision; ...
     max(abs(eps_reach)), max(abs(eps_self_collision)), max(abs(eps_env_collision)); ...
     t_reach_train, t_self_collision_train, t_env_collision_train; ...
     t_reach_test, t_self_collision_test, t_env_collision_test];

result_path = "./results/interpolation_training.mat";
save(result_path, 'T');