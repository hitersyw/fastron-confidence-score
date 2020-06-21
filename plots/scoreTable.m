% Generate interpolation vs. regression optimization table; 
close all; clear
rng(0);
init;

datetime = "16_04_2020_19";
file_spec = "./dvrkData/cone/pose/pose_%s_n%d_%s_%s_validated.mat";
n_sparse = 64;
n_dense = 4096;
psm1 = "psm1";
psm2 = "psm2";

%% SVR
load(sprintf(file_spec, datetime, n_sparse, "weightedSVR", psm1));
T_svr_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime, n_sparse, "weightedSVR", psm2));
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

%% Coarse to fine method
load(sprintf(file_spec, datetime, n_dense, "coarse2fine", psm1));
T_coarse2fine_psm1 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

load(sprintf(file_spec, datetime,  n_dense, "coarse2fine", psm2));
T_coarse2fine_psm2 = validated_score([6, 9, 12, 15]); % TODO: fix this hardcode;

%% Metrics; 
metrics = {'reachability','self-collision','env-collision', 'combined'};
row_names = {'Dataset-sparse', 'Interpolation', 'Regression', 'Coarse2fine', 'Dataset-dense (GT)'};
T_psm1 = [T_dataset_sparse_psm1; T_interpolation_psm1; T_svr_psm1; T_coarse2fine_psm1; T_dataset_dense_psm1]; 
T_psm2 = [T_dataset_sparse_psm2;T_interpolation_psm2; T_svr_psm2; T_coarse2fine_psm2; T_dataset_dense_psm2]; 

% rescale
% T_psm1 = T_psm1(:, 1:3);
% T_psm2 = T_psm2(:, 1:3);
T_psm1(:, 4) = T_psm1(:, 4) / 3;
T_psm2(:, 4) = T_psm2(:, 4) / 3;

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
