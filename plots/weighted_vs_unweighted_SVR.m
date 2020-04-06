close all; clear

load("./results/svr_training_weighted.mat");
T_weighted = T(1:2, :)

load("./results/svr_training.mat");
T_unweighted = T(1:2, :)

row_names = {"Reachability mse", "Reachability max difference", ...
              "Self-collision mse", "Self-collision max difference", ...
              "Env-collision mse", "Env-collision max difference"};
metrics = [T_weighted(:), T_unweighted(:)];
matrix2latex(metrics, "./results/weighted_vs_unweighted.tex", ...
    'rowLabels', row_names, 'columnLabels', {'Weighted', 'Unweighted'},...
    'alignment', 'c', 'format', '%.4f');