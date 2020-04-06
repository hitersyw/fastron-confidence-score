close all; clear
rng(0);
init;

% Data generation
n_sparse = 288; 
n_dense = 1053;

path = "~/workspace/arclab/fastron-confidence-score/dvrk_data/cone/log/workspace_x0.1_0.3_y0.1_0.3_two_arms/";

%% Load Data Generation time;

% Load sparse dataset;
t_sparse = load_data_generation_time(path, true, n_sparse);
% Load dense dataset; 
t_dense = load_data_generation_time(path, true, n_dense);

%% Load training time;
% Interpolation;
T_interpolation_psm1 = load(sprintf("./results/interpolation_training_n%d_%s.mat", n_sparse, "psm1"));
t_interpolation_train_psm1 = T_interpolation_psm1.T(3, :); 

T_interpolation_psm2 = load(sprintf("./results/interpolation_training_n%d_%s.mat", n_sparse, "psm2"));
t_interpolation_train_psm2 = T_interpolation_psm2.T(3, :);
t_interpolation_train = t_interpolation_train_psm1 + t_interpolation_train_psm2;

% SVR;
T_svr_psm1 = load(sprintf("./results/svr_training_weighted_%d_%s.mat", n_sparse, "psm1"));
t_svr_train_psm1 = T_svr_psm1.T(3, :); 

T_svr_psm2 = load(sprintf("./results/svr_training_weighted_%d_%s.mat", n_sparse, "psm2"));
t_svr_train_psm2 = T_svr_psm2.T(3, :); 

t_svr_train = t_svr_train_psm1 + t_svr_train_psm2;

%% Load Optimization time;
% Interpolation
load(sprintf("./results/optimal_pose_interpolation_n%d_%s.mat", n_sparse, "psm1")); 
t_interpolation_op_psm1 = T_interpolation_optimization(end);

load(sprintf("./results/optimal_pose_interpolation_n%d_%s.mat", n_sparse, "psm2")); 
t_interpolation_op_psm2 = T_interpolation_optimization(end);

t_interpolation_op = t_interpolation_op_psm1 + t_interpolation_op_psm2;

% SVR
load(sprintf("./results/optimal_pose_svr_n%d_%s.mat", n_sparse, "psm1"));
t_svr_op_psm1 = T_svr_op(end);

load(sprintf("./results/optimal_pose_svr_n%d_%s.mat", n_sparse, "psm2"));
t_svr_op_psm2 = T_svr_op(end); 

t_svr_op = t_svr_op_psm1 + t_svr_op_psm2;
%% Combine all into a score; 
row_names = {'Sparse Interpolation', 'Sparse Dataset', 'Sparse-Regression', 'Dense-Dataset'};
column_names = {'Data Generation', 'Model Fitting', 'Optimization', 'Total'};

interpolation = [sum(t_sparse), sum(t_interpolation_train), sum(t_interpolation_op)];
sparse_model_free = [sum(t_sparse), 0, 0];
regression = [sum(t_sparse), sum(t_svr_train), sum(t_svr_op)];
dense_model_free = [sum(t_dense), 0, 0];

T = [interpolation; sparse_model_free; regression; dense_model_free];
T = [T, sum(T, 2)];
file_name = './results/interpolation_vs_regression_timing.tex';
matrix2latex(T, file_name, 'rowLabels', row_names, ...
  'columnLabels', column_names, ...
  'alignment', 'c', 'format', '%.2f');