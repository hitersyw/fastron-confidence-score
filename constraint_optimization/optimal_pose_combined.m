%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";
output_path = base_dir + "pose/poses_%s.csv";

reachability_dataset = 'reachability_score';
self_collision_dataset = 'collision_score';
n = 2925;

shuffle = false;

%% Load Dataset
[X_reach_train, y_reach_train, X_reach_test, y_reach_test] = ...
    load_dvrk2(input_path, reachability_dataset, n, shuffle, false);
[X_collision_train, y_collision_train, X_collision_test, y_collision_test] ...
    = load_dvrk2(input_path, self_collision_dataset, n, shuffle, false);
X_train = X_reach_train;
X_test = X_reach_test;

%% Normalize the dataset;
xmax = max([X_train; X_test]);
xmin = min([X_train; X_test]);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Train the models
self_collision_mdl = trainModel(X_train, y_collision_train, X_test, y_collision_test, 'svr');
reachability_mdl = trainModel(X_train, y_reach_train, X_test, y_reach_test, 'svr');

%% Predict output and find the max of labels;
X_uniform = (xmax - xmin).*rand(100000,size(X_test,2)) + xmin;
X_uniform = scale_input(X_uniform);

y_uniform_reachability = predict(reachability_mdl, X_uniform);
y_uniform_collision = predict(self_collision_mdl, X_uniform);

y_max_reachability = max(y_uniform_reachability);   
y_max_collision = max(y_uniform_collision);

scale_output_collision = @(y) y./y_max_collision;
scale_output_reach = @(y) y./y_max_reachability;

%% Calculate losses
y_collision_pred = predict(self_collision_mdl, X_test);
l_collision = y_collision_test - y_collision_pred;
l_scaling_collision = y_collision_test - scale_output_collision(y_collision_pred);
fprintf("Collision MSE Loss before scaling: %.4f, after scaling: %.4f \n", ...
    (l_collision'*l_collision)/size(y_collision_test, 1), ...
    (l_scaling_collision'*l_scaling_collision)/size(y_collision_test, 1));

y_reach_pred = predict(reachability_mdl, X_test);
l_reach = y_reach_test - y_reach_pred;
l_scaling_reach = y_reach_test - scale_output_reach(y_reach_pred);
fprintf("Reachability MSE Loss before scaling: %.4f, after scaling: %.4f \n", ...
    (l_reach'*l_reach)/size(y_reach_test, 1), ...
    (l_scaling_reach'*l_scaling_reach)/size(y_reach_test, 1));

%% Find optimal poses
n_init = 100;
X_init = (xmax - xmin).* rand(n_init, size(xmax, 1)) + xmin;
% x0 = [-1.0826, -0.3033, 0.6599, -1.309];
z = 0.6599;

X = zeros(size(X_init, 1), size(X_init, 2) + 3);
for i=1:size(X, 1)
    x0 = X_init(i, :);
    x = find_pose_combined(x0, xmin, xmax, self_collision_mdl, reachability_mdl,...
        scale_input, scale_output_collision, scale_output_reach);
    
    % print scores; 
    reachability_score = scale_output_reach(predict(reachability_mdl, scale_input(x)));
    collision_score = scale_output_collision(predict(self_collision_mdl, scale_input(x)));
    fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s; Collision score: %s \n",...
            x, reachability_score, collision_score);
    % fprintf("Position: [%.3f, %.3f, %.3f]; Predicted self-collision score: %s; Actual: %s\n", x, self_collision_score);
    X(i, :) = [x(1:2), z, x(3), reachability_score, collision_score];
end

path = sprintf(output_path, 'combined');
writematrix(X, path);




