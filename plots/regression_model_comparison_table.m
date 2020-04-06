close all;
init;
format shortE; 


load('./results/regression_models.mat');
file_name = './results/regression_model_comparison.tex';
rowLabels = T1.Properties.RowNames;
columnLabels = T1.Properties.VariableNames;
matrix2latex(T1.Variables, file_name, 'rowLabels', rowLabels, 'columnLabels', columnLabels, 'alignment', 'c', 'format', '%.2e');