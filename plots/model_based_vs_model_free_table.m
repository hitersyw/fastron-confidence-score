close all; clear;
init; 

n = 288;
% Load model-based table; 
load(sprintf("./results/svr_training_weighted_%d_%s.mat", n, "psm1"));
T_psm1 = T;

load(sprintf("./results/svr_training_weighted_%d_%s.mat", n, "psm2"));
T_psm2 = T;

% Load model-based optimization table;
load(sprintf("./results/optimal_pose_svr_n%d_%s.mat", n, "psm1"));
fCount_psm1 = T_svr_op(9);

load(sprintf("./results/optimal_pose_svr_n%d_%s.mat", n, "psm2"));
fCount_psm2 = T_svr_op(9);

%% Create table for model-based and model-free comparison
% Use only SVR for comparison with model-free method; 
% n_init = 100;
num_checks = fCount_psm1 + fCount_psm2;
model_free = [2.2, 0.443, 0.214];
model_based = T_psm1(4, :) + T_psm2(4, :);

metrics = zeros(numel(model_free) * 2, 2);
metrics([1,3,5],1) = model_based';
metrics([2,4,6],1) = model_free';
metrics(:, 2) = metrics(:, 1) * num_checks;
row_names = {'Model-based reachability', 'Model-free reachability', 'Model-based self-collision',...
    'Model-free self-collision', 'Model-based env-collision', 'Model-free env-collision'};

matrix2latex(metrics, "./results/model_based_vs_model_free.tex", 'rowLabels', row_names, ...
    'columnLabels', {'Time per Check', 'Time during optimization'}, ...
    'alignment', 'c');