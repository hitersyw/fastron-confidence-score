close all; clear
run('./init.m');
rng(0);

input_path = base_dir + "log/%s_n%d.mat";
output_path = base_dir + "pose/poses_%s.csv";

dataset = 'reachability_score';
n = 2925;
[X_train, y_train, X_test, y_test] = load_dvrk(input_path, dataset, n, false);
[max_ytrain, max_idx_train] = max(y_train);
[max_ytest, max_idx_test] = max(y_test);
fprintf("The maximum of y_train is: %.4f; position: [%.4f, %.4f, %.4f]\n", max_ytrain, X_train(max_idx_train, :));
fprintf("The maximum of y_test is: %.4f; position: [%.4f, %.4f, %.4f]\n", max_ytest, X_train(max_idx_test, :));

X = [X_train; X_test];
y = [y_train; y_test];
F = scatteredInterpolant(X, y, 'linear', 'nearest');

ub = max([X_train; X_test], [], 1); % upper bound of each column;
lb = min([X_train; X_test], [], 1); % lower bound

%x0 = (ub + lb) / 2;
x0 = [-0.9826, -0.2533, -1.2090];
% x0 = [-0.9326, -0.0033, -1.4090];
z = 0.6599;

% x0 = x0([1 2 4]);
x = find_pose_interpolation(x0, lb, ub, F);
reachability_score = F(x);
% self_collision_score = F(x);
% fprintf("Position: [%.3f, %.3f, %.3f]; Collision score is: %s\n", x, self_collision_score);
fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %.3f\n", x, reachability_score);
% fprintf("Position: [%.3f, %.3f, %.3f]; Predicted self-collision score: %s; Actual: %s\n", x, self_collision_score);
