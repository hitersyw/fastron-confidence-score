% Generate interpolation vs. regression optimization table; 
close all; clear
rng(0);
init;

datetime = "02_03_2020_14";
file_spec = "./dvrk_data/cone/pose/pose_%s_n%d_%s_%s_validated.mat";
n_sparse = 288;
n_dense = 1053;
psm1 = "psm1";
psm2 = "psm2";

%% SVR
load(sprintf(file_spec, datetime, n_sparse, "SVR_weighted", psm1));
T_svr_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime, n_sparse, "SVR_weighted", psm2));
T_svr_psm2 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

%% Interpolation
load(sprintf(file_spec, datetime, n_sparse, "interpolation", psm1));
T_interpolation_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime, n_sparse, "interpolation", psm2));
T_interpolation_psm2 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

%% Sparse dataset
load(sprintf(file_spec, datetime, n_sparse, "dataset", psm1));
T_dataset_sparse_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime, n_sparse, "dataset", psm2));
T_dataset_sparse_psm2 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

%% Dense dataset
load(sprintf(file_spec, datetime, n_dense, "dataset", psm1));
T_dataset_dense_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime,  n_dense, "dataset", psm2));
T_dataset_dense_psm2 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

%% Metrics; 
metrics = {'reachability','self-collision','env-collision'};
row_names = {'Interpolation', 'Dataset-sparse', 'Regression', 'Dataset-dense'};
T_psm1 = [T_interpolation_psm1; T_dataset_sparse_psm1; T_svr_psm1; T_dataset_dense_psm1]; 
T_psm2 = [T_interpolation_psm2; T_dataset_sparse_psm2; T_svr_psm2; T_dataset_dense_psm2]; 

% rescale
T_psm1 = T_psm1(:, 1:3);
T_psm2 = T_psm2(:, 1:3);

file_spec = './results/interpolation_vs_regression_%s.tex';
matrix2latex(T_psm1, sprintf(file_spec, psm1), 'rowLabels',row_names, ...
  'columnLabels', metrics, ...
  'alignment', 'c', 'format', '%.2f');

matrix2latex(T_psm2, sprintf(file_spec, psm2), 'rowLabels',row_names, ...
  'columnLabels', metrics, ...
  'alignment', 'c', 'format', '%.2f');

%% Bar graph

% PSM 1; 
figure();
b = bar(T_psm1);
set(gca,'xticklabel', row_names);
set(gca,'xticklabel', row_names);
ylim([0, 1.2]);

b(1).FaceColor = 'r';
b(2).FaceColor = 'g';
b(3).FaceColor = 'b';
grid on; 
legend(metrics);

title("PSM 1");

% PSM 2; 
figure();
b = bar(T_psm2);
set(gca,'xticklabel', row_names);
set(gca,'xticklabel', row_names);
ylim([0, 1.2]);

b(1).FaceColor = 'r';
b(2).FaceColor = 'g';
b(3).FaceColor = 'b';
grid on; 
legend(metrics);

title("PSM 2");
