format shortE;
load("svr_training.mat");

row_names = {'MSE','Max Absolute difference','Training Time','Test-time'};
col_names = {'Reachability', 'Self Collision', 'Environment Collision'};

result_table = array2table(T, 'VariableNames', col_names, 'RowNames', row_names)
table2latex(result_table, './results/SVR_training.tex');

