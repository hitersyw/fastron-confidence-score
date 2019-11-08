clear; close all;
rng(0);

%% Hyper-parameters;
n = 1125; 
dataset = 'reachability_score';
maxIter = 5000;
g = 40;

%% Load collision_score data
score_dict = load(sprintf('/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/ConfidenceScore/dvrk_data/%s_n%d.mat', dataset, n));
score = getfield(score_dict, dataset);
X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

p_train = 0.8;
idx = randperm(n); % shuffle the dataset;
X = X(idx, :);
X_train = X(1:n*p_train, :);
y_train = y(1:n*p_train);

X_test = X(n*p_train+1:n, :);
y_test = y(n*p_train+1:n);

K = rbf(X_train, X_train, g);
M = size(K,2);

fprintf("Size of training set: %d; test set: %d", size(X_train,1), size(X_test,1));

%% Define model parameters
OPTIONS = SB2_UserOptions('iterations', maxIter,...
                          'diagnosticLevel', 2,...
						  'monitor', 10);
SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

%% Fit Relevance Vector Machine (Sparse Bayesian Model) on the data;
tic();
[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
    SparseBayes('Gaussian', K, y_train, OPTIONS, SETTINGS);
w_infer						= zeros(M,1);
w_infer(PARAMETER.Relevant)	= PARAMETER.Value;
rvm_train = toc();
fprintf("Number of Relevance points: %d\n",size(PARAMETER.Relevant, 1));

% Compute MSE Loss;
tic();
K_test = rbf(X_test, X_train, g);
y_pred = K_test(:,PARAMETER.Relevant)*w_infer(PARAMETER.Relevant); 
rvm_test = toc();
mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
eps = y_pred - y_test;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);
fprintf("Training time: %s\nTest time per data point: %s\n", rvm_train, rvm_test/size(X_test, 1));

