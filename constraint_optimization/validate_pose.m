close all; clear
rng(0);
init;

% model_path = "./dvrk_data/saved_model/11_02_2020_15_n1053.mat";
model_path = "./dvrk_data/saved_model/16_02_2020_13_n2925.mat";
output_path = base_dir + "test";

%% load saved workspace and models
load(model_path);
% x = [-1.1426, -0.3733, -1.6090];
x = [-1.0826, -0.3033, -1.4090];
z = 0.6599;

reachability_score = clip((predict(reachability_mdl, scale_input(x))),0.0001);
collision_score = clip((predict(self_collision_mdl, scale_input(x))),0.0001);
env_collision_score = clip(predict(env_collision_mdl, scale_input(x)),0.0001);
% reachability_score = F_reach(x);
% collision_score = F_self_collision(x);
% env_collision_score = F_env_collision(x);

scores = [reachability_score, collision_score, env_collision_score];
fprintf("Position: [%.3f, %.3f, %.3f]\nReachability score is: %s\n" ...
        + "Self-collision score: %s\nEnv-collision score: %s\n"...
        + "Combined score: %s", ...
        x, reachability_score, collision_score, env_collision_score, ...
        reachability_score + collision_score + env_collision_score);

X_out = [x(1:2), z, x(3), scores, sum(scores)];

% Write output
if ~exist(output_path, 'dir')
   mkdir(output_path)
end
    
path = output_path + "/poses_single.csv";
writematrix(X_out, path);
