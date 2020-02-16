%% Parameters;
close all; clear
rng(0);
init;

input_path = base_dir + "log/%s_n%d.mat";

reachability_dataset = 'reachability_score_smote';
n_original = 640;
n_max = 10; % top n poses to show from the dataset;
shuffle = true; % whether to shuffle the dataset; 
p_test = 0.2;

%% Load Dataset
[X_reach, y_reach] = load_dvrk3(input_path, reachability_dataset, n_original, false);
X = X_reach;
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
    y_reach = y_reach(idx);
end

% Test set; 
X_test = X(1:ceil(n*p_test), :);
y_reach_test = y_reach(1:ceil(n*p_test));

% Training set;
X_train = X(ceil(n*p_test+1):n, :);
y_reach_train = y_reach(ceil(n*p_test+1):n);

tic();
reachability_mdl = trainSVR(X_train, y_reach_train, X_test, y_reach_test, false, 22.6207, 0.3, 0.0001);
t_reach_train = toc();