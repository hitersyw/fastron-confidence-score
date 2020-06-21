close all; clear
rng(0);
init;

% Data generation
n_sparse = 64; 
n_dense = 4096;

path = "~/workspace/arclab/fastron-confidence-score/dvrkData/cone/log/workspace_x0.3_0.3_y0.3_0.3_two_arms_ik/";

%% Load Data Generation time;

% Load dense dataset; 
t_dense = loadDataGenerationTime(path, n_dense);

% Load sparse dataset 
t_sparse = t_dense / (n_dense / n_sparse);

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

%% Load coarse2fine time
load(sprintf("./results/optimal_pose_coarse2fine_n%d_%s.mat", n_dense, "psm1"));
num_checks_psm1 = T_coarse2fine_search(end);
load(sprintf("./results/optimal_pose_coarse2fine_n%d_%s.mat", n_dense, "psm2")); 
num_checks_psm2 = T_coarse2fine_search(end);
t_coarse2fine = (num_checks_psm1 + num_checks_psm2) * (0.5 * t_dense / n_dense); 

%% Combine all into a score; 
row_names = {'Sparse Model-free', 'Sparse Interpolation', 'Sparse-Regression', 'Dense Model-free', 'Coarse2Fine'};
column_names = {'Data Generation', 'Model Fitting', 'Optimization', 'Total'};

interpolation = [sum(t_sparse), sum(t_interpolation_train), sum(t_interpolation_op)];
sparse_model_free = [sum(t_sparse), 0, 0];
regression = [sum(t_sparse), sum(t_svr_train), sum(t_svr_op)];
dense_model_free = [sum(t_dense), 0, 0];
coarse2fine = [sum(t_coarse2fine), 0, 0];

T = [sparse_model_free; interpolation; regression; dense_model_free; coarse2fine];
T = [T, sum(T, 2)];
file_name = './results/interpolation_vs_regression_timing.tex';
matrix2latex(T, file_name, 'rowLabels', row_names, ...
  'columnLabels', column_names, ...
  'alignment', 'c', 'format', '%.2f');

%% Bar graph

% PSM 1; 
figure();
b = bar(T(:, 4));
set(gca,'xticklabel', row_names);
ylabel('seconds');
% b(1).FaceColor = 'r';
% b(2).FaceColor = 'g';
% b(3).FaceColor = 'b';
grid on; 

title("Timing statistics");