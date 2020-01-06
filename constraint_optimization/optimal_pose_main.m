close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";
output_path = base_dir + "pose/poses_%s.csv";

dataset = 'collision_score';
n = 2925;

%% Load Dataset
[X_train, y_train, X_test, y_test] = load_dvrk2(input_path, dataset, n, false);
[max_ytrain, max_idx_train] = max(y_train);
[max_ytest, max_idx_test] = max(y_test);
fprintf("The maximum of y_train is: %.2f; position: [%.3f, %.3f, %.3f]\n", max_ytrain, X_train(max_idx_train, :));
fprintf("The maximum of y_test is: %.2f; position: [%.3f, %.3f, %.3f]\n", max_ytest, X_test(max_idx_test, :));

%% Normalize the dataset;
xmax = max([X_train; X_test]);
xmin = min([X_train; X_test]);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Train the models;
% self_collision_mdl = trainModel(X_train, y_train, X_test, y_test, 'svr');
mdl = trainModel(X_train, y_train, X_test, y_test, 'svr');

%% Predict output and find the max of y;
X_uniform = (xmax - xmin).*rand(100000,size(X_test,2)) + xmin;
X_uniform = scale_input(X_uniform);
y_uniform = predict(mdl, X_uniform);
y_max = max(y_uniform);
scale_output = @(y) y./y_max;

y_pred = predict(mdl, X_test);
l = y_test - y_pred;
l_scaling = y_test - scale_output(y_pred);
fprintf("SVR MSE Loss before scaling: %.4f, after scaling: %.4f \n", (l'*l)/size(y_test, 1), (l_scaling'*l_scaling)/size(y_test, 1));

%% Find optimal poses
n_init = 10;
X_init = (xmax - xmin).* rand(n_init, size(xmax, 1)) + xmin;
% x0 = [-1.0826, -0.3033, 0.6599, -1.309];
z = 0.6599;

X = zeros(size(X_init, 1), size(X_init, 2) + 1);
for i=1:size(X, 1)
    x0 = X_init(i, :);
    x = find_pose(x0, xmin, xmax, mdl, scale_input, scale_output);
    % x = find_pose(x0, lb, ub, self_collision_mdl);
    score = scale_output(predict(mdl, scale_input(x)));
    % self_collision_score = predict(self_collision_mdl, x);
    fprintf("Position: [%.3f, %.3f, %.3f]; Score is: %s\n", x, score);
    % fprintf("Position: [%.3f, %.3f, %.3f]; Predicted self-collision score: %s; Actual: %s\n", x, self_collision_score);
    X(i, :) = [x(1:2), z, x(3)];
end

path = sprintf(output_path, '11_19');
writematrix(X, path);
% TODO: output it to a csv file for validation on the dVRK scene;
