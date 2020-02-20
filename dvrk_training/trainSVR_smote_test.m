%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";
n_total = 1053;
% n_original = 640;
% result_path = sprintf('./results/svr_training_%d.mat', n_total);

% training_set = sprintf('reachability_score_%d', n_total);
training_set = 'reachability_score';
test_set = 'reachability_score';
n_test = 1053;
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 

%% Load Dataset
[X_train, y_train] = load_dvrk3(input_path, training_set, n_total, false);
n = size(X_train, 1);

[X_test, y_test] = load_dvrk3(input_path, test_set, n_test, false);
%% Normalize the dataset;
xmax = max(X_train);
xmin = min(X_train);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;

X_train = scale_input(X_train);
X_test = scale_input(X_test);

%% Shuffle and divide up the dataset;
if shuffle
    % shuffle the dataset;
    idx = randperm(n); 
    X_train = X_train(idx, :);
    y_train = y_train(idx);
end

tic();
weights = quadratic(1, 0.5, y_train);
reachability_mdl = trainWeightedSVR(X_train, y_train, X_test, y_test, false, weights, 0.4641, 0.46416, 0.0051434);
t_reach_train = toc();

%% Compute test loss
tic();
y_pred = predict(reachability_mdl, X_test);
l_reach = y_pred - y_test;
mse_reach = l_reach' * l_reach / size(X_test, 1);
t_reach_test = toc() / size(X_test, 1);

abs_loss_grid = abs(l_reach);


%% Plot error of model fitting
figure('Position', [327 87 800 800]);
ind = abs_loss_grid > 0.05;

% reachability;
cm = jet(256);
subplot(1,2,1)
colorScatter3(X_test(ind,1),X_test(ind,2),...
    X_test(ind,3),abs_loss_grid(ind), cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Error plot");
    
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');

subplot(1,2,2)
colorScatter3(X_test(:,1),X_test(:,2),...
    X_test(:,3),y_pred, cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Reachability plot");
    
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Prediction plot");
sgtitle("Model-based scores grid");

%% Save model
T = [mse_reach, max(abs(l_reach)), t_reach_train, t_reach_test];
save(result_path, 'T');