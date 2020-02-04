%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";
output_path = base_dir + "pose/poses_%s.csv";

reachability_dataset = 'reachability_score';
self_collision_dataset = 'self_collision_score';
environment_collision_dataset = 'env_collision_score';
n = 2925;
p_dataset = 0.2; % percentage of poses to initialize from the datset;
n_init = 50;

%% Load Dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n, false);
[X_self_collision, y_self_collision] = load_dvrk3(input_path, self_collision_dataset, n, false);
[X_env_collision, y_env_collision] = load_dvrk3(input_path, environment_collision_dataset, n, false);

assert(all(X_reach == X_self_collision, 'all'));
assert(all(X_reach == X_env_collision, 'all'));
X = X_reach;

%% Extract maximum score poses from dataset
combined_raw_score = y_reach + y_self_collision + y_env_collision;
top_n = 20;
[max_poses, max_scores] = max_score_poses(X, [y_reach, y_self_collision, y_env_collision, combined_raw_score], n_init*p_dataset);
display("Displaying maximum poses and their scores");
display([max_poses, max_scores]);

%% Normalize the dataset;
xmax = max(X);
xmin = min(X);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X = scale_input(X);

%% Shuffle and divide up the dataset;
shuffle = true;
p_test = 0.2;
if shuffle
    % shuffle the dataset;
    idx = randperm(n); 
    X = X(idx, :);
    y_self_collision = y_self_collision(idx);
    y_env_collision = y_env_collision(idx);
    y_reach = y_reach(idx);
end

% Test set; 
X_test = X(1:ceil(n*p_test), :);
y_reach_test = y_reach(1:ceil(n*p_test));
y_self_collision_test = y_self_collision(1:ceil(n*p_test));
y_env_collision_test = y_env_collision(1:ceil(n*p_test));

% Training set;
X_train = X(ceil(n*p_test+1):n, :);

y_reach_train = y_reach(ceil(n*p_test+1):n);
y_self_collision_train = y_self_collision(ceil(n*p_test+1):n);
y_env_collision_train = y_env_collision(ceil(n*p_test+1):n);

%% Train the models
self_collision_mdl = trainSVR(X_train, y_self_collision_train, X_test, y_self_collision_test);
env_collision_mdl = trainSVR(X_train, y_env_collision_train, X_test, y_env_collision_test);
reachability_mdl = trainSVR(X_train, y_reach_train, X_test, y_reach_test);

%% Predict output and find the max of labels;
X_uniform = (xmax - xmin).*rand(10000,size(X_test,2)) + xmin;
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

% scaling functions for output;
scale_output_env_collision = @(y) (y - y_min_env_collision)./(y_max_env_collision - y_min_env_collision);
scale_output_collision = @(y) (y - y_min_collision)./(y_max_collision - y_min_collision);
scale_output_reach = @(y) (y - y_min_reachability)./(y_max_reachability - y_min_reachability);

%% Calculate losses
y_self_collision_pred = predict(self_collision_mdl, X_test);
l_self_collision = y_self_collision_test - y_self_collision_pred;
l_scaling_self_collision = y_self_collision_test - scale_output_collision(y_self_collision_pred);
fprintf("Self-Collision MSE Loss before scaling: %.4f, after scaling: %.4f \n", ...
    (l_self_collision'*l_self_collision)/size(y_self_collision_test, 1), ...
    (l_scaling_self_collision'*l_scaling_self_collision)/size(y_self_collision_test, 1));

y_env_collision_pred = predict(env_collision_mdl, X_test);
l_env_collision = y_env_collision_test - y_env_collision_pred;
l_scaling_env_collision = y_env_collision_test - scale_output_collision(y_env_collision_pred);
fprintf("Env-Collision MSE Loss before scaling: %.4f, after scaling: %.4f \n", ...
    (l_env_collision'*l_env_collision)/size(y_env_collision_test, 1), ...
    (l_scaling_env_collision'*l_scaling_env_collision)/size(y_env_collision_test, 1));

y_reach_pred = predict(reachability_mdl, X_test);
l_reach = y_reach_test - y_reach_pred;
l_scaling_reach = y_reach_test - scale_output_reach(y_reach_pred);
fprintf("Reachability MSE Loss before scaling: %.4f, after scaling: %.4f \n", ...
    (l_reach'*l_reach)/size(y_reach_test, 1), ...
    (l_scaling_reach'*l_scaling_reach)/size(y_reach_test, 1));

%% Plot the 3D space of raw self-collision, environment collision, and reachability scores.
thres = 0.5;
cm = jet(256);
figure('Position', [327 87 800 800]);
subplot(2,2,1)
colorScatter3(X(y_reach>thres,1),X(y_reach>thres,2),...
              X(y_reach>thres,3),y_reach(y_reach>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Reachability > .5");

subplot(2,2,2)
colorScatter3(X(y_self_collision>thres,1),X(y_self_collision>thres,2),...
              X(y_self_collision>thres,3),y_self_collision(y_self_collision>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Self-collision > .5");

subplot(2,2,3)
colorScatter3(X(y_env_collision>thres,1),X(y_env_collision>thres,2),...
              X(y_env_collision>thres,3),y_env_collision(y_env_collision>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Env-collision > .5");

subplot(2,2,4) 
colorScatter3(X(y_reach>thres & y_self_collision > thres & y_env_collision > thres,1),...
              X(y_reach>thres & y_self_collision > thres & y_env_collision > thres, 2),...
              X(y_reach>thres & y_self_collision > thres & y_env_collision > thres,3), ...
              combined_raw_score(y_reach >thres & y_self_collision >thres & y_env_collision > thres),...
              cm);
view([153 58]); axis square; grid on;
title("All > .5");
xlabel('X');
ylabel('Y');
zlabel('\theta');
sgtitle("Raw scores");

%% Plot the 3D grid space of collision and reachability scores produced with models;
y_reach_grid = scale_output_reach(predict(reachability_mdl, X));
figure('Position', [327 87 800 800]);

% reachability;
cm = jet(256);
subplot(2,2,1)
colorScatter3(X(y_reach_grid>thres,1),X(y_reach_grid>thres,2),...
    X(y_reach_grid>thres,3),y_reach_grid(y_reach_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Reachability > .5");
    
% Self Collision
y_collision_grid = scale_output_collision(predict(self_collision_mdl, X));
subplot(2,2,2)
colorScatter3(X(y_collision_grid>thres ,1),X(y_collision_grid>thres,2),...
    X(y_collision_grid>thres,3),y_collision_grid(y_collision_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Self-collision > .5");

% Env Collision
y_env_collision_grid = predict(env_collision_mdl, X);
subplot(2,2,3)
colorScatter3(X(y_env_collision_grid>thres ,1),X(y_env_collision_grid>thres,2),...
    X(y_env_collision_grid>thres,3),y_env_collision_grid(y_env_collision_grid>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Env-collision > .5");

% Combined Score
combined_score_grid = y_collision_grid + y_reach_grid + y_env_collision_grid;
subplot(2,2,4)
colorScatter3(X(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres,1),...
              X(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres, 2),...
              X(y_collision_grid>thres & y_reach_grid > thres & y_env_collision_grid > thres,3), ...
              combined_score_grid(y_collision_grid>thres & y_reach_grid >thres & y_env_collision_grid > thres),...
              cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("All > .5");
sgtitle("Model-based scores grid");

%% Plot the 3D uniform space of collision and reachability scores produced with models;
y_reach_uniform = scale_output_reach(predict(reachability_mdl, X_uniform));
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
title("Reachability > .5");
    
% Self Collision
y_collision_uniform = scale_output_collision(predict(self_collision_mdl, X_uniform));
subplot(2,2,2)
colorScatter3(X_uniform(y_collision_uniform>thres ,1),X_uniform(y_collision_uniform>thres,2),...
    X_uniform(y_collision_uniform>thres,3),y_collision_uniform(y_collision_uniform>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Self-collision > .5");

% Env Collision
y_env_collision_uniform = predict(env_collision_mdl, X_uniform);
subplot(2,2,3)
colorScatter3(X_uniform(y_env_collision_uniform>thres ,1),X_uniform(y_env_collision_uniform>thres,2),...
    X_uniform(y_env_collision_uniform>thres,3),y_env_collision_uniform(y_env_collision_uniform>thres), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Env-collision > .5");

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
title("All > .5");
sgtitle("Model-based scores uniform");


%% Find optimal poses
% n_init = 100;
% X_init = (xmax - xmin).* rand(n_init, size(xmax, 1)) + xmin;
X_init = (xmax - xmin).* rand(n_init * (1 - p_dataset), size(xmax, 1)) + xmin;
X_init = [X_init; max_poses];
z = 0.6599;

X_out = zeros(size(X_init, 1), size(X_init, 2) + 5);
max_score = 0;
fCount = 0;
total_time = 0;

for i=1:size(X_out, 1)
    x0 = X_init(i, :);
    tic();
    [x,fval,exitflag, output] = find_pose_combined(x0, xmin, xmax, self_collision_mdl, reachability_mdl,...
        env_collision_mdl, scale_input, scale_output_collision, scale_output_reach);
    total_time = total_time + toc();
    
    % print scores; 
    reachability_score = clip(scale_output_reach(predict(reachability_mdl, scale_input(x))),0.0001);
    collision_score = clip(scale_output_collision(predict(self_collision_mdl, scale_input(x))),0.0001);
    env_collision_score = clip(predict(env_collision_mdl, scale_input(x)),0.0001);
    scores = [reachability_score, collision_score, env_collision_score];
    fprintf("Position: [%.3f, %.3f, %.3f]; Reachability score is: %s;" ...
            + "Self-collision score: %s; Env-collision score: %s\n",...
            x, reachability_score, collision_score, env_collision_score);
    fprintf("Score: %.3f, functionCount: %d\n", -fval, output.funcCount);
    max_score = max(-fval, max_score);
    fCount = fCount + output.funcCount;
    % fprintf("Position: [%.3f, %.3f, %.3f]; Predicted self-collision score: %s; Actual: %s\n", x, self_collision_score);
    X_out(i, :) = [x(1:2), z, x(3), scores, sum(scores)];
end

fprintf("Maximum score is: %.2f\n", max_score);
fprintf("Average number of function counts per initialization: %.2f\n", fCount / size(X_out, 1));
fprintf("Average optimization time per initialization: %.4f\n", total_time / size(X_out, 1));

% Threshold;
thres = 0.5; 
% X_out = X_out(all(X_out(:, 5:7)>thres, 2), :);

% Dedupe poses;
[C, IA, IC] = uniquetol(X_out(:,[1,2,4]), 0.05, 'ByRows', true);
X_out = X_out(IA, :);

% Sort; 
X_out = sortrows(X_out, size(X_out, 2), 'descend');

% Write output
path = sprintf(output_path, 'combined');
writematrix(X_out, path);
writematrix([max_poses(:, 1:2), z * ones(size(max_poses, 1), 1), max_poses(:, 3), max_scores], ...
    sprintf(output_path, 'dataset'));
