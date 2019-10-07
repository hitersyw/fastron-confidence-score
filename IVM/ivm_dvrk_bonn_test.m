clear; close all;
rng(0)

%% Load dataset
dir = "/home/jamesdi1993/workspace/fastron_experimental/fastron_vrep/log";
sample_file_spec = dir + "/joint_angle_sample_X_n%d_arm%d.csv";
collision_label_spec = dir + "/collision_state_y_n%d_arm%d.csv";
arm = 1;
n = 10000;

sample_file = sprintf(sample_file_spec, n, arm);
collision_label_file = sprintf(collision_label_spec, n, arm);
X = dlmread(sample_file,',',0,0);
y = dlmread(collision_label_file,',',0,0);
y = (y+3)/2; % input is -1 and 1; convert to 1 and 2;

training_p = 0.8;
validation_p = 0.1;
test_p = 0.1;

% train, validation and test splits;
X_train = X(1:training_p * n, :);
y_train = y(1:training_p * n, :);
X_holdout = X(training_p*n+1:(validation_p + training_p)*n, :);
y_holdout = y(training_p*n+1:(validation_p + training_p)*n, :);
X_test = X((validation_p + training_p)*n+1:n, :);
y_test = y((validation_p + training_p)*n+1:n, :);

%% Train IVM
params      = init;
params.Nadd = inf;

% with kernel and regularization parameter
params.output   = 1;
params.sigma    = 20; % corresponding to gamma in rbf;
params.lambda   = 5;
params.epsilon = 0.001;

% learn model
data.phi = X_train'; % m x n
data.phit = X_test'; % n x 1
data.c = y_train; % n x 1
data.ct = y_test; % t x 1

% learn and predict model
profile on;
result = ivm(data, params);
profile off;