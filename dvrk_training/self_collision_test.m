%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";

self_collision_dataset = 'self_collision_score';
n_original = 1053;
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 
p_test = 0.2;

%% Load Dataset
[X_self_collision, y_self_collision] = load_dvrk3(input_path, self_collision_dataset, n_original, false);
X = X_self_collision;
n = size(X, 1);

%% Normalize the dataset;
xmax = max(X);
xmin = min(X);
% scale_input = @(x) x;
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X = scale_input(X);

%% Shuffle and divide up the dataset;
if shuffle
    % shuffle the dataset;
    idx = randperm(n); 
    X = X(idx, :);
    y_self_collision = y_self_collision(idx);
end

% Test set; 
X_test = X(1:ceil(n*p_test), :);
y_self_collision_test = y_self_collision(1:ceil(n*p_test));

% Training set;
X_train = X(ceil(n*p_test+1):n, :);
y_self_collision_train = y_self_collision(ceil(n*p_test+1):n);

% weights = quadratic(1, 0.5, y_self_collision_train); 
weights = gaussian_weight(0.5, 0.33, y_self_collision_train);
% weights = ones(numel(y_self_collision_train));

tic();
self_collision_mdl = trainWeightedSVR(X_train, y_self_collision_train, X_test, y_self_collision_test, true, weights, 33.8527, 1.1889, 0.048989);
t_self_collision_train = toc();

%% Plot the error of self-collision training
y_pred = predict(self_collision_mdl, X);
error = abs(y_self_collision - y_pred); 

cm = jet(256);
figure();
thres = 0.05; 

subplot(1,2,1);
colorScatter3(X(error > thres,1),X(error > thres,2),...
    X(error > thres,3), error(error > thres) , cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Self-collision error > 0.1");

subplot(1,2,2);
colorScatter3(X(:,1),X(:,2),...
    X(:,3), y_self_collision, cm);
view([153 58]); axis square; grid on;
xlabel('X');
ylabel('Y');
zlabel('\theta');
title("Self-collision prediction");
