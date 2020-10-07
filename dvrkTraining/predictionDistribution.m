% Plot the distribution of test samples for SVR, MLP and RVM.
close all; clear
rng(0);
init;

g = 50; 

% Change the input directories here.
input_path = base_dir + 'cone/samples/workspace_x0.1_0.3_y0.1_0.3/%s_n%d.mat';

dataset = 'reachability_score';
n = 1053;
[X_train, y_train, X_test, y_test] = loadDvrk(input_path, dataset, n, false, false, 0);
size = size([y_train; y_test], 1);

%% SVR;
svrMdl = svrTrain(X_train, y_train, X_test, y_test);
y_svr = predict(svrMdl, X_train);

%% RVM;
K_train = rbf(X_train, X_train, g);
K_test = rbf(X_test, X_train, g);
w_rvm = rvmTrain(K_train, y_train, K_test, y_test);
y_rvm = K_train * w_rvm;

%% MLP
h = [64,64];            % two hidden layers with 64 and 64 neurons
lambda = 0.0001;        % Regularized Loss; 
[model, L] = mlpReg(X_train',y_train',h,lambda);
y_mlp = mlpRegPred(model, X_test')';
e_mlp = y_mlp - y_test;
fprintf("MSE for MLP: %.2f\n", e_mlp'*e_mlp / numel(y_test));

close all;
figure; clf;
h1 = histogram(y_train, 10, 'Normalization','probability','BinWidth', 0.1);
hold on;
h2 = histogram(y_svr, 10, 'Normalization', 'probability','BinWidth', 0.1);
hold on;
h3 = histogram(y_rvm, 10, 'Normalization', 'probability','BinWidth', 0.1);
hold on;
h4 = histogram(y_mlp, 10, 'Normalization', 'probability','BinWidth', 0.1);
legend([h1, h2, h3, h4], {'Original', 'SVR', 'RVM', 'MLP'},  'location','northeast');
title(sprintf('%s, n:%d', strrep(dataset, '_', ' '), size));