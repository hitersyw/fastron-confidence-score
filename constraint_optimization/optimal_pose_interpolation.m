% Need to run dvrk_interpolation.m first to save the interpolated models;
% Save the workspace produced by dvrk_interpolation.m, and then change the 
% model_path variable to load the variable.

%% Parameters;
close all; clear
rng(0);
init;
format shortE;

% Load model;
n = 252;
n_init = 100;
arm = "psm2"; 
datetime = "25_03_2020_10";
model_path = sprintf("./dvrk_data/saved_model/%s_n%d_interpolation_%s.mat", datetime, n, arm);
load(model_path);

% arm = "psm1"; % TODO: this is because when we load model_path, psm1 is included and we need to reset it. 


% Define output path;
result_path = sprintf("./results/optimal_pose_interpolation_n%d_%s.mat", n, arm);
output_path = base_dir + "pose";
output_name = sprintf("pose_%s_n%d_interpolation_%s.csv", datetime, n, arm);

%% Find optimal poses
x0 = [0, 0, 0];
z = 0.6599;

% optimization
tic();
[x,fval,exitflag, output] = find_pose_interpolation(x0, n_init, -ones(1, 3), ones(1, 3), ...
    F_reach, F_self_collision, F_env_collision);
total_time = toc();

%% Evaluate the scores of the found pose; 
reachability_score = clip(F_reach(x), 0.00001);
collision_score = clip(F_self_collision(x), 0.00001);
env_collision_score = clip(F_env_collision(x), 0.00001);

scores = [reachability_score, collision_score, env_collision_score];
fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
        + "Self-collision score: %s; Env-collision score: %s\n",...
        x, reachability_score, collision_score, env_collision_score);
fprintf("Score: %.3f, functionCount: %d\n", -fval, output.funcCount);

% scale x back to original scale;
x = (x + 1) / 2; % first scale back to [0, 1];
x = xmin + x.*(xmax - xmin);

max_score = -fval;
fCount = output.funcCount;

% [x, y, z, theta, reach, self_collision, combined_score, # calls, Time];
X_out = [x(1:2), z, x(3), scores, sum(scores)];

fprintf("Maximum score is: %.2f\n", sum(scores));
fprintf("Average number of function counts per initialization: %.2f\n", fCount / n_init);
fprintf("Average optimization time per initialization: %.4f\n", total_time);
% Write timing and accuracy result for plotting; 

% Write output for validation in VREP; 
if ~exist(output_path, 'dir')
   mkdir(output_path)
end

path = output_path + "/" + output_name;
fprintf("Writing to output file: %s", path);
writematrix(X_out, path);
    
T_interpolation_optimization = [X_out, fCount, total_time];
save(result_path, 'T_interpolation_optimization');
