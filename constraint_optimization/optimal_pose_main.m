close all; clear
rng(0);

base_path = '/home/jamesdi1993/workspace/arclab/fastron_experimental/fastron_vrep/constraint_analysis/log/%s_n%d.mat';
output_path = "/home/jamesdi1993/workspace/arclab/fastron_experimental/fastron_vrep/constraint_analysis/pose/poses_%s.csv";

% self_collision_mdl = train(base_path, 'collision_score', );
reachability_mdl = trainModel(base_path, 'reachability_score', 1125, 'svr');

x0 = [-1.0826, -0.3033, 0.6599, -1.309];
z = x0(3);

x0 = x0([1 2 4]);
x_offset = [0.25, 0.45];
y_offset = [0.3, 0.4];
theta_offset = [-0.3, 0.3];

lb = x0 + [x_offset(1), y_offset(1), theta_offset(1)];
ub = x0 + [x_offset(2), y_offset(2), theta_offset(2)];
x = find_pose(x0, lb, ub, reachability_mdl);

reachability_score = predict(reachability_mdl, x);
fprintf("The reachability score is: %s", reachability_score);

path = sprintf(output_path, '11_19')
X = [x(1:2), z, x(3)];
writematrix(X, output_path);
% TODO: output it to a csv file for validation on the dVRK scene;
