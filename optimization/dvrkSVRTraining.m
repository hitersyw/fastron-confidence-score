% Script for training self-collision, environment-collision and
% reachability models on the DVRK datasets. After training the models and
% plotting, save the workspace variables, which can then be loaded for
% optimization. 
%% Parameters;
close all; clear
rng(0);
init;

% TODO: move this into a common configuration file; 
arm = "psm2";
base_dir = base_dir + 'cone/';
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";
input_path = base_dir + "log/" + data_dir;

input_spec = input_path + "%s_n%d.mat";
reachability_dataset = "reachability_score" + "_" + arm;
self_collision_dataset = "self_collision_score" + "_" + arm;
env_collision_dataset = "env_collision_score" + "_" + arm;
use_fastron = true;
n = 252;
n_test = 1872;

n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 
tol = 0.125;
tune_params = true;
result_path = sprintf("./results/svr_training_weighted_%d_%s.mat", n, arm);

%% Load Training Dataset
[X_reach_train, y_reach_train] = loadDvrk2(input_spec, reachability_dataset, n, use_fastron,false);
[X_self_collision_train, y_self_collision_train] = loadDvrk2(input_spec, self_collision_dataset, n, use_fastron, false);
[X_env_collision_train, y_env_collision_train] = loadDvrk2(input_spec, env_collision_dataset, n, use_fastron, false);

% safety check;
assert(all(X_reach_train == X_self_collision_train, 'all'));
assert(all(X_reach_train == X_env_collision_train, 'all'));
X_train = X_reach_train;

%% Load workspace limit and safety check
run(input_path + "workspace_limit.m");
% xmax = max(X_train);
% xmin = min(X_train);

if arm == "psm1"
    xmax = xmax_psm1;
    xmin = xmin_psm1;
elseif arm == "psm2"
    xmax = xmax_psm2;
    xmin = xmin_psm2;
end
assert(all(max(X_train) <= xmax + tol));
assert(all(min(X_train) >= xmin - tol));

%% Load Test Dataset and safety check;
[X_reach_test, y_reach_test] = loadDvrk2(input_spec, reachability_dataset, n_test, use_fastron,false);
[X_self_collision_test, y_self_collision_test] = loadDvrk2(input_spec, self_collision_dataset, n_test, use_fastron,false);
[X_env_collision_test, y_env_collision_test] =  loadDvrk2(input_spec, env_collision_dataset, n_test, use_fastron,false);
% Safety Check;
assert(all(X_reach_test == X_self_collision_test, 'all'));
assert(all(X_reach_test == X_env_collision_test, 'all'));
assert(all(max(X_reach_test) <= xmax + tol));
assert(all(min(X_reach_test) >= xmin - tol));

X_test = X_reach_test;

%% Extract maximum score poses from dataset
combined_raw_score = y_reach_train + y_self_collision_train + y_env_collision_train;
top_n = 10;
[max_poses, max_scores] = maxScorePoses(X_train, [y_reach_train, y_self_collision_train, y_env_collision_train, combined_raw_score], n_max);
display("Displaying maximum poses and their scores");
display([max_poses, max_scores]);

%% Normalize the dataset;
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Train the models
tic(); 
reach_weights = quadratic(1, 0.5, 0.5, y_reach_train);
self_collision_weights = quadratic(1, 0.5, 0.5, y_self_collision_train);
env_collision_weights = quadratic(1, 0.5, 0.5, y_self_collision_train);

tic();
reachability_mdl = trainWeightedSVR(X_train, y_reach_train, X_test, y_reach_test, tune_params, ...
    reach_weights, 3.1999e+00, 5.3248e-01, 7.8300e-03);
t_reach_train = toc();

tic();
self_collision_mdl = trainWeightedSVR(X_train, y_self_collision_train, X_test, y_self_collision_test, tune_params, ...
    self_collision_weights, 1.7468e+01, 1.8649e+00, 1.9789e-02);
t_self_collision_train = toc();

tic();
env_collision_mdl = trainWeightedSVR(X_train, y_env_collision_train, X_test, y_env_collision_test, tune_params, ...
    env_collision_weights, 1.0247e-03, 3.8539e-01, 7.7020e-04);
t_env_collision_train = toc();


%% Predict output and find the max of labels;
X_uniform = (xmax - xmin).*rand(5000,size(X_test,2)) + xmin;
X_uniform = scale_input(X_uniform);

% labels
y_uniform_reachability = predict(reachability_mdl, X_uniform);
y_uniform_collision = predict(self_collision_mdl, X_uniform);
y_uniform_env_collision = predict(env_collision_mdl, X_uniform);

% max of labels
y_max_reachability = max(y_uniform_reachability);   
y_max_collision = max(y_uniform_collision);
y_max_env_collision = max(y_uniform_env_collision);

% min of labels
y_min_reachability = min(y_uniform_reachability);   
y_min_collision = min(y_uniform_collision);
y_min_env_collision = min(y_uniform_env_collision);

% max of combined_labels
y_max_combined = max(y_uniform_reachability + y_uniform_collision + y_uniform_env_collision);
y_min_combined = min(y_uniform_reachability + y_uniform_collision + y_uniform_env_collision);

fprintf("Maximum combined score from uniform sampling: %.4f\n", y_max_combined);

% scaling functions for output;
scale_output_env_collision = @(y) (y - y_min_env_collision)./(y_max_env_collision - y_min_env_collision);
scale_output_collision = @(y) (y - y_min_collision)./(y_max_collision - y_min_collision);
scale_output_reach = @(y) (y - y_min_reachability)./(y_max_reachability - y_min_reachability);

%% Calculate losses
tic();
y_self_collision_pred = predict(self_collision_mdl, X_test);
t_self_collision_test = toc() / size(X_test, 1); 

l_self_collision = y_self_collision_test - y_self_collision_pred;
l_scaling_self_collision = y_self_collision_test - scale_output_collision(y_self_collision_pred);
fprintf("Self-Collision MSE Loss before scaling: %.4f, after scaling: %.4f, Maximum diff: %.4f\n", ...
    (l_self_collision'*l_self_collision)/size(y_self_collision_test, 1), ...
    (l_scaling_self_collision'*l_scaling_self_collision)/size(y_self_collision_test, 1), ...
    max(abs(l_self_collision))...
    );

tic();
y_env_collision_pred = predict(env_collision_mdl, X_test);
t_env_collision_test = toc() / size(X_test, 1); 

l_env_collision = y_env_collision_test - y_env_collision_pred;
l_scaling_env_collision = y_env_collision_test - scale_output_collision(y_env_collision_pred);
fprintf("Env-Collision MSE Loss before scaling: %.4f, after scaling: %.4f, Maximum diff: %.4f\n", ...
    (l_env_collision'*l_env_collision)/size(y_env_collision_test, 1), ...
    (l_scaling_env_collision'*l_scaling_env_collision)/size(y_env_collision_test, 1), ...
    max(abs(l_env_collision))...
    );

tic();
y_reach_pred = predict(reachability_mdl, X_test);
t_reach_test = toc() / size(X_test, 1); 

l_reach = y_reach_test - y_reach_pred;
l_scaling_reach = y_reach_test - scale_output_reach(y_reach_pred);
fprintf("Reachability MSE Loss before scaling: %.4f, after scaling: %.4f, Maximum diff: %.4f\n", ...
    (l_reach'*l_reach)/size(y_reach_test, 1), ...
    (l_scaling_reach'*l_scaling_reach)/size(y_reach_test, 1), ...
    max(abs(l_reach))...
    );

%% Write out performance for plotting;
mse_reach = l_reach' * l_reach / size(y_reach_test, 1);
mse_self_collision = l_self_collision' * l_self_collision / size(y_reach_test, 1);
mse_env_collision = l_env_collision' * l_env_collision / size(y_reach_test, 1);

T = [mse_reach, mse_self_collision, mse_env_collision; ...
     max(abs(l_reach)), max(abs(l_self_collision)), max(abs(l_env_collision)); ...
     t_reach_train, t_self_collision_train, t_env_collision_train; ...
     t_reach_test, t_self_collision_test, t_env_collision_test];

save(result_path, 'T');

%% Plot the 3D space of raw self-collision, environment collision, and reachability scores.
thres = 0.5;
cm = jet(256);
figure('Position', [327 87 800 800]);
subplot(2,2,1)
colorScatter3(X_train(y_reach_train>thres,1),X_train(y_reach_train>thres,2),...
              X_train(y_reach_train>thres,3),y_reach_train(y_reach_train>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Reachability > %s", thres));

subplot(2,2,2)
colorScatter3(X_train(y_self_collision_train>thres,1),X_train(y_self_collision_train>thres,2),...
              X_train(y_self_collision_train>thres,3),y_self_collision_train(y_self_collision_train>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Self-collision > %s", thres));

subplot(2,2,3)
colorScatter3(X_train(y_env_collision_train>thres,1),X_train(y_env_collision_train>thres,2),...
              X_train(y_env_collision_train>thres,3),y_env_collision_train(y_env_collision_train>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Env-collision > %s", thres));

subplot(2,2,4) 
colorScatter3(X_train(y_reach_train>thres & y_self_collision_train > thres & y_env_collision_train > thres,1),...
              X_train(y_reach_train>thres & y_self_collision_train > thres & y_env_collision_train > thres, 2),...
              X_train(y_reach_train>thres & y_self_collision_train > thres & y_env_collision_train > thres,3), ...
              combined_raw_score(y_reach_train >thres & y_self_collision_train >thres & y_env_collision_train > thres),...
              cm);
view([153 58]); axis square; grid on;
title("All > .5");
xlabel('X');
ylabel('Y');
zlabel('\theta');
sgtitle("Raw scores");

%% Plot the 3D grid space of collision and reachability scores produced with models;
y_reach_grid = scale_output_reach(predict(reachability_mdl, X_test));
figure('Position', [327 87 800 800]);

% reachability;
cm = jet(256);
subplot(2,2,1)
colorScatter3(X_test(y_reach_grid>thres,1),X_test(y_reach_grid>thres,2),...
    X_test(y_reach_grid>thres,3),y_reach_grid(y_reach_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Reachability > %.1f", thres));
    
% Self Collision
y_collision_grid = scale_output_collision(predict(self_collision_mdl, X_test));
subplot(2,2,2)
colorScatter3(X_test(y_collision_grid>thres ,1),X_test(y_collision_grid>thres,2),...
    X_test(y_collision_grid>thres,3),y_collision_grid(y_collision_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Self-collision > %.1f", thres));

% Env Collision
y_env_collision_grid = predict(env_collision_mdl, X_test);
subplot(2,2,3)
colorScatter3(X_test(y_env_collision_grid>thres ,1),X_test(y_env_collision_grid>thres,2),...
    X_test(y_env_collision_grid>thres,3),y_env_collision_grid(y_env_collision_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Env-collision > %.1f", thres));

% Combined Score
combined_score_grid = y_collision_grid + y_reach_grid + y_env_collision_grid;
subplot(2,2,4)
colorScatter3(X_test(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres,1),...
              X_test(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres, 2),...
              X_test(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres,3), ...
              combined_score_grid(y_collision_grid>thres & y_reach_grid >thres & y_env_collision_grid > thres),...
              cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("All > %.1f", thres));
sgtitle("Model-based scores grid");

%% Plot the 3D uniform space of collision and reachability scores produced with models;
y_reach_uniform = predict(reachability_mdl, X_uniform);
figure('Position', [327 87 800 800]);
% reachability;
cm = jet(256);
subplot(2,2,1)
colorScatter3(X_uniform(y_reach_uniform>thres,1),X_uniform(y_reach_uniform>thres,2),...
    X_uniform(y_reach_uniform>thres,3),y_reach_uniform(y_reach_uniform>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Reachability > %.1f", thres));
    
% Self Collision
y_collision_uniform = predict(self_collision_mdl, X_uniform);
subplot(2,2,2)
colorScatter3(X_uniform(y_collision_uniform>thres ,1),X_uniform(y_collision_uniform>thres,2),...
    X_uniform(y_collision_uniform>thres,3),y_collision_uniform(y_collision_uniform>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Self-collision > %.1f", thres));

% Env Collision
y_env_collision_uniform = predict(env_collision_mdl, X_uniform);
subplot(2,2,3)
colorScatter3(X_uniform(y_env_collision_uniform>thres ,1),X_uniform(y_env_collision_uniform>thres,2),...
    X_uniform(y_env_collision_uniform>thres,3),y_env_collision_uniform(y_env_collision_uniform>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("Env-collision > %.1f", thres));

% AND three conditions
combined_score_uniform = y_collision_uniform + y_reach_uniform + y_env_collision_uniform;
subplot(2,2,4)
colorScatter3(X_uniform(y_collision_uniform>thres & y_reach_uniform > thres & y_env_collision_uniform > thres,1),...
              X_uniform(y_collision_uniform>thres & y_reach_uniform > thres & y_env_collision_uniform > thres, 2),...
              X_uniform(y_collision_uniform>thres & y_reach_uniform > thres & y_env_collision_uniform > thres,3), ...
              combined_score_uniform(y_collision_uniform>thres & y_reach_uniform >thres & y_env_collision_uniform > thres),...
              cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title(sprintf("All > %.1f", thres));
sgtitle("Model-based scores uniform");