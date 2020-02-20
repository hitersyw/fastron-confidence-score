% Need to run optimal_pose_training.m first to save the trained models;
% Save the workspace, then change the model_path variable to load model.

%% Parameters;
close all; clear
rng(0);
init;
format shortE;

model_path = "./dvrk_data/saved_model/16_02_2020_17_n640_svr_weighted.mat";
output_path = base_dir + "pose/";
output_name = "pose_16_02_2020_17_SVR_weighted.csv"

%% load saved workspace and models
load(model_path);

%% Find optimal poses
% x0 = [-1.0826, -0.3033, -1.309]; % maximum reacability;
x0 = [-1.1426, -0.3733, 0.9769]; % maximum combined score from the dataset; 
% x0 = [-1.1726, -0.3433, -1.5590];
x0 = scale_input(x0);
z = 0.6599;

% optimization
tic();
[x,fval,exitflag, output] = find_optimal_pose_global_search_normalized(x0, -ones(1, 3), ones(1, 3), ...
    self_collision_mdl, reachability_mdl,...
    env_collision_mdl,  scale_output_collision, scale_output_reach);
total_time = toc();

%% Evaluate the scores of the found pose; 

% print scores; 
reachability_score = clip(predict(reachability_mdl, x), 0.00001);
collision_score = clip(predict(self_collision_mdl, x), 0.00001);
env_collision_score = clip(predict(env_collision_mdl, x), 0.00001);
scores = [reachability_score, collision_score, env_collision_score];

% scale x back to original scale;
x = (x + 1) / 2; % first scale back to [0, 1];
x = xmin + x.*(xmax - xmin);

fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
        + "Self-collision score: %s; Env-collision score: %s\n",...
        x, reachability_score, collision_score, env_collision_score);
fprintf("Score: %.3f, functionCount: %d\n", -fval, output.funcCount);

max_score = -fval;
fCount = output.funcCount;
X_out = [x(1:2), z, x(3), scores, sum(scores)];

fprintf("Maximum score is: %.2f\n", sum(scores));
fprintf("Average number of function counts per initialization: %.2f\n", fCount);
fprintf("Average optimization time per initialization: %.4f\n", total_time);

% Write output for validation;
if ~exist(output_path, 'dir')
   mkdir(output_path)
end
    
path = output_path + "/" + output_name; 
writematrix(X_out, path);

% Write output for plotting;
T_svr = [X_out, fCount, total_time];
result_path = "./results/optimal_pose_svr.mat";
save(result_path, 'T_svr');
