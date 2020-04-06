% Need to run optimal_pose_training.m first to save the trained models;
% Save the workspace, then change the model_path variable to load model.

%% Parameters;
close all; clear
rng(0);
init;
format shortE;

%% load saved workspace and models
n = 252;
n_init = 100;
arm = "psm2"; 
datetime = "25_03_2020_11";
model_path = sprintf("./dvrkData/saved_model/%s_n%d_svr_weighted_%s.mat", datetime, n, arm);
load(model_path);

%% Define output path; 
result_path = sprintf("./results/optimal_pose_svr_n%d_%s.mat", n, arm);
output_path = base_dir + "pose/";
output_name = sprintf("pose_%s_n%d_weightedSVR_%s.csv", datetime, n, arm);

%% Find optimal poses
% x0 = [-1.0826, -0.3033, -1.309]; % maximum reacability;
% x0 = [-1.1426, -0.3733, 0.9769]; % maximum combined score from the dataset; 
x0 = [-1.1726, -0.3433, -1.5590];
x0 = scale_input(x0);
% x0 = [0, 0, 0];
z = 0.6599;

% optimization
tic();
[x,fval,exitflag, output] = findPoseGlobalSearchNormalized(x0, n_init, -ones(1, 3), ones(1, 3), ...
     self_collision_mdl, reachability_mdl,...
     env_collision_mdl);

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
X_out = [x(1:2), z, x(3), scores, max_score];

fprintf("Maximum score is: %.2f\n", sum(scores));
fprintf("Average number of function counts per initialization: %.2f\n", fCount / n_init);
fprintf("Average optimization time per initialization: %.4f\n", total_time);

% Write output for validation;
if ~exist(output_path, 'dir')
   mkdir(output_path)
end
    
path = output_path + output_name; 
writematrix(X_out, path);

% Write output for plotting;
T_svr_op = [X_out, fCount, total_time];
save(result_path, 'T_svr_op');
