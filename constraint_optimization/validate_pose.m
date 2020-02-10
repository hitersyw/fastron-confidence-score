close all; clear
rng(0);
init;

model_path = "./dvrk_data/saved_model/08_02_2020_15.mat";
output_path = base_dir + "test";

%% load saved workspace and models
load(model_path);
x = [-1.1426, -0.3733, -1.6090];
z = 0.6599;

reachability_score = clip(scale_output_reach(predict(reachability_mdl, scale_input(x))),0.0001);
collision_score = clip(scale_output_collision(predict(self_collision_mdl, scale_input(x))),0.0001);
env_collision_score = clip(predict(env_collision_mdl, scale_input(x)),0.0001);
scores = [reachability_score, collision_score, env_collision_score];
fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
        + "Self-collision score: %s; Env-collision score: %s\n",...
        x, reachability_score, collision_score, env_collision_score);

X_out = [x(1:2), z, x(3), scores, sum(scores)];

% Write output
if ~exist(output_path, 'dir')
   mkdir(output_path)
end
    
path = output_path + "/poses_single.csv";
writematrix(X_out, path);
