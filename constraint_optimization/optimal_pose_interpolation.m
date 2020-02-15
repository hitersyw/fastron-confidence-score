% Need to run dvrk_interpolation.m first to save the interpolated models;
% Save the workspace produced by dvrk_interpolation.m, and then change the 
% model_path variable to load the variable.

%% Parameters;
close all; clear
rng(0);
init;
format shortE;

model_path = "./dvrk_data/saved_model/14_02_2020_22_n1053_interpolation.mat";
output_path = base_dir;

%% load saved workspace and models
load(model_path);

%% Find optimal poses
% x0 = [-1.0826, -0.3033, -1.409]; % maximum reacability;
% x0 = [-1.1426, -0.3733, -1.6090]; % maximum combined score from the dataset; 
x0 = [-1.1726, -0.3433, -1.5590];  
z = 0.6599;

% optimization
tic();
[x,fval,exitflag, output] = find_pose_interpolation(x0, xmin, xmax, ...
    F_reach, F_self_collision, F_env_collision);
total_time = toc();

%% Evaluate the scores of the found pose; 
reachability_score = F_reach(x);
collision_score = F_self_collision(x);
env_collision_score = F_env_collision(x);

scores = [reachability_score, collision_score, env_collision_score];
fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
        + "Self-collision score: %s; Env-collision score: %s\n",...
        x, reachability_score, collision_score, env_collision_score);
fprintf("Score: %.3f, functionCount: %d\n", -fval, output.funcCount);

max_score = -fval;
fCount = output.funcCount;

% scale x back to original scale;
x = (x + 1) / 2; % first scale back to [0, 1];
x = xmin + x.*(xmax - xmin);

% [x, y, z, theta, reach, self_collision, combined_score, # calls, Time];
X_out = [x(1:2), z, x(3), scores, sum(scores)];

fprintf("Maximum score is: %.2f\n", max_score);
fprintf("Average number of function counts per initialization: %.2f\n", fCount);
fprintf("Average optimization time per initialization: %.4f\n", total_time);

% Write output for validation in VREP; 
if ~exist(output_path, 'dir')
   mkdir(output_path)
end

path = output_path + "/poses_combined_normalized.csv";
writematrix(X_out, path);
    
% Write timing and accuracy result for plotting; 
T_interpolation = [X_out, fCount, total_time];
result_path = "./results/optimal_pose_interpolation.mat";
save(result_path, 'T_interpolation');
