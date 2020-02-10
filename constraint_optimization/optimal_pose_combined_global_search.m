% Need to run optimal_pose_training.m first to save the trained models;

%% Parameters;
close all; clear
rng(0);
init;

model_path = "./dvrk_data/saved_model/08_02_2020_15.mat";
output_path = base_dir + "test";

%% load saved workspace and models
load(model_path);

%% Find optimal poses
% x0 = [-1.0826, -0.3033, -1.409]; % maximum reacability;
% x0 = [-1.1426, -0.3733, -1.6090]; % maximum combined score from the dataset; 
x0 = [-1.1726, -0.3433, -1.5590];  
z = 0.6599;

% optimization
tic();
[x,fval,exitflag, output] = find_optimal_pose_global_search(x0, xmin, xmax, ...
    self_collision_mdl, reachability_mdl,...
    env_collision_mdl, scale_input, scale_output_collision, scale_output_reach);
total_time = toc();

%% Evaluate the scores of the found pose; 
reachability_score = clip(scale_output_reach(predict(reachability_mdl, scale_input(x))),0.0001);
collision_score = clip(scale_output_collision(predict(self_collision_mdl, scale_input(x))),0.0001);
env_collision_score = clip(predict(env_collision_mdl, scale_input(x)),0.0001);
scores = [reachability_score, collision_score, env_collision_score];
fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
        + "Self-collision score: %s; Env-collision score: %s\n",...
        x, reachability_score, collision_score, env_collision_score);
fprintf("Score: %.3f, functionCount: %d\n", -fval, output.funcCount);

max_score = -fval;
fCount = output.funcCount;
% fprintf("Position: [%.3f, %.3f, %.3f]; Predicted self-collision score: %s; Actual: %s\n", x, self_collision_score);
X_out = [x(1:2), z, x(3), scores, sum(scores)];

fprintf("Maximum score is: %.2f\n", max_score);
fprintf("Average number of function counts per initialization: %.2f\n", fCount);
fprintf("Average optimization time per initialization: %.4f\n", total_time);

% Write output
if ~exist(output_path, 'dir')
   mkdir(output_path)
end
    
path = output_path + "/poses_combined.csv";
writematrix(X_out, path);
