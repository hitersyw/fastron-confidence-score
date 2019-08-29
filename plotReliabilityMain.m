%% Setup
clear; close all;

dataset = "CSpace1";

% Hyper-parameters for Fastron learning;
iterMax = 5000;
g = 1.;
beta = 1.;
Smax = 2000;
num_bins = 10;
lr = 0.001;
eps = 0.0001;

x_min = [-4, -4];
x_max = [4, 4];
r = 1;

% training_file = "./circle_training.csv";
% test_file = "./circle_test.csv";
training_file_spec = "./data/%s_training.csv";
validation_file_spec = "./data/%s_validation.csv";
test_file_spec = "./data/%s_test.csv";
title_spec = "%s (%f)";

training_file = sprintf(training_file_spec, dataset);
validation_file = sprintf(validation_file_spec, dataset);
test_file = sprintf(test_file_spec, dataset);
[X_train, y_train] = loadData2(training_file);
% y_train(y_train == -1, :)=0;

[X_holdout, y_holdout] = loadData2(validation_file);
% y_holdout(y_holdout == -1, :)=0;

[X_test, y_test] = loadData2(test_file);
% y_test(y_test == -1, :)=0;

%% Plot test set;
figure('NumberTitle', 'off');
subplot(2,3,1);
x1 = X_test(y_test == 1, :);
x2 = X_test(y_test == -1, :);

scatter(x1(:, 1), x1(:, 2), 8, 'r', 'filled');
hold on;
scatter(x2(:, 1), x2(:, 2), 8, 'b', 'filled');
title("Original");

%% Fastron learning with RBF; 
[a, F, K, iter]=trainFastron(X_train, y_train, @rbf, iterMax, Smax, beta, g);
K_test_rbf = rbf(X_test, X_train, g); % n x m; 
F_test_rbf = K_test_rbf*a;
x_pos = X_test(F_test_rbf >= 0, :);
x_neg = X_test(F_test_rbf < 0, :);
p_rbf = 1./(1 + exp(-F_test_rbf)); % without calibration;
y_pred_fastron = sign(F_test_rbf);

subplot(2,3,2);
scatter(x_pos(:, 1), x_pos(:,2), 8, 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 8, 'b', 'filled');
title(sprintf(title_spec, "Fastron RBF", sum(sign(F_test_rbf)==y_test)/size(y_test,1)));

% results(1,1,:) = (sign(F_test_rbf) == 1 && y_test == 1);
% results(1,2,:) = (sign(F_test_rbf) == 1 && y_test ~= 1);
% results(1,3,:) = (sign(F_test_rbf) ~= 1 && y_test ~= 1);
% results(1,4,:) = (sign(F_test_rbf) ~= 1 && y_test == 1);
%% Load output from LR
log_reg_file_spec = "./sgd_%s.json";
log_reg_file=sprintf(log_reg_file_spec,dataset);
fid = fopen(log_reg_file);
raw = fread(fid); 
str = char(raw'); 
fclose(fid); 
values = jsondecode(str);
w_lr = values.coef;
b_lr = values.intercept;

lr_test_output = values.test_output;
y_pred_lr_test = lr_test_output(:,1);
p_lr_test = lr_test_output(:, 3);
% lr_validation_output = values.validation_output;
% y_pred_lr_validation = data(:,1);
% p_lr_validation = data(:, 3);

x_pos = X_test(y_pred_lr_test == 1, :);
x_neg = X_test(y_pred_lr_test == 0, :);

subplot(2,3,3);
scatter(x_pos(:, 1), x_pos(:,2), 8, 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 8, 'b', 'filled');
title(sprintf(title_spec, "Log Regression", sum(y_pred_lr_test * 2-1==y_test)/size(y_test,1)));

%% Load output from NN
mlp_file_spec = "./mlp_%s.json";
mlp_file = sprintf(mlp_file_spec, dataset);
fid = fopen(mlp_file);
raw = fread(fid); 
str = char(raw'); 
fclose(fid); 
values = jsondecode(str);
w_mlp = values.coef;
b_mlp = values.intercept;
data = values.test_output;
y_pred_mlp = data(:,1);
p_mlp = data(:, 3);

x_pos = X_test(y_pred_mlp == 1, :);
x_neg = X_test(y_pred_mlp == 0, :);

subplot(2,3,4);
scatter(x_pos(:, 1), x_pos(:,2), 8, 'r','filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 8, 'b','filled');
title(sprintf(title_spec, "MLP", sum(y_pred_mlp * 2-1==y_test)/size(y_test,1)));
%% Load output from bagged trees
bagging_reg_file_spec = "./bagged_trees_%s.json";
bagging_reg_file=sprintf(bagging_reg_file_spec,dataset);
fid = fopen(bagging_reg_file);
raw = fread(fid); 
str = char(raw'); 
fclose(fid); 
values = jsondecode(str);

bagging_test_output = values.test_output;
y_pred_bagging_test = bagging_test_output(:,1);
p_bagging = bagging_test_output(:, 2);

x_pos = X_test(y_pred_bagging_test == 1, :);
x_neg = X_test(y_pred_bagging_test == 0, :);

subplot(2,3,5);
scatter(x_pos(:, 1), x_pos(:,2), 8, 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 8, 'b', 'filled');
title(sprintf(title_spec, "Bagged Trees", sum(y_pred_bagging_test * 2-1==y_test)/size(y_test,1)));

%% IVM learning
lambda = 5;
useUnbiasedVersion = false;
K = rbf(X_train, X_train, g);

if useUnbiasedVersion
    [a_ivm, S, idx] = ivmTrain(X_train, y_train, K, lambda);
else
    [a_ivm, S, idx] = ivmTrain2(X_train, y_train, K, lambda);
end

if useUnbiasedVersion
    F_test_IVM = rbf(X_test, S, g)*a_ivm;
else
    F_test_IVM = rbf(X_test, S, g)*a_ivm(1:end-1) + a_ivm(end);
end
p_ivm = 1./(1 + exp(-F_test_IVM));
y_pred_ivm = sign(F_test_IVM); 

x_pos = X_test(p_ivm >= 0.5, :);
x_neg = X_test(p_ivm < 0.5, :);

subplot(2,3,6);
scatter(x_pos(:, 1), x_pos(:,2), 8, 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 8, 'b', 'filled');
title(sprintf(title_spec, "IVM", sum(sign(F_test_IVM)==y_test)/size(y_test,1)));

%% Calibration Fastron
K_holdout = rbf(X_holdout, X_train, g); % n x m; 
F_holdout = K_holdout*a;
[A_fastron, B_fastron] = trainPlattScaling(F_holdout, y_holdout, iterMax, eps, lr);

p_rbf_calibrated = 1./(1 + exp(A_fastron.*F_test_rbf + B_fastron));

%% Calibration Kernel LogReg;
F_holdout_lr = K_holdout * w_lr' + b_lr;
[A_lr, B_lr] = trainPlattScaling(F_holdout_lr, y_holdout, iterMax, eps, lr);
p_lr_calibrated = 1./(1 + exp(A_lr.*F_test_rbf + B_lr));

%% Calibration NN;
F_holdout_mlp = feedforward(X_holdout, w_mlp, b_mlp);
[A_mlp, B_mlp] = trainPlattScaling(F_holdout_mlp, y_holdout, iterMax, eps, lr);

F_test_mlp = feedforward(X_test, w_mlp, b_mlp);
p_mlp_calibrated = 1./(1 + exp(A_mlp.* F_test_mlp + B_mlp));

%% Calibration IVM;
if useUnbiasedVersion
    F_holdout_IVM = rbf(X_test, S, g)*a_ivm;
else
    F_holdout_IVM = rbf(X_test, S, g)*a_ivm(1:end-1) + a_ivm(end);
end
[A_ivm, B_ivm] = trainPlattScaling(F_holdout, y_holdout, iterMax, eps, lr);
p_ivm_calibrated = 1./(1 + exp(A_ivm * F_test_IVM + B_ivm));

%% Reliability diagrams; 
% plotReliability([p_rbf(:) p_lr_test(:) p_mlp(:) p_bagging(:)],...
%     y_test, num_bins, ["Fastron RBF" "LogRegression" "MLP" "Bagged Trees"]); 
% 
% plotReliability([p_rbf_calibrated(:) p_lr_calibrated(:) p_mlp_calibrated(:)],...
%     y_test, num_bins, ["Fastron RBF Calibrated" "LogRegression Calibrated" "MLP Calibrated"]); 
plotReliability2([p_rbf(:) p_rbf_calibrated(:) p_lr_test(:) p_lr_calibrated(:) p_mlp(:) p_mlp_calibrated(:) p_ivm(:) p_ivm_calibrated(:)], ...
    y_test, num_bins, ["Fastron" "LogReg" "MLP" "IVM"]); 

%% Probability plots
[X2,Y2] = meshgrid(linspace(x_min(1), x_max(1),100), linspace(x_min(2),x_max(2),100));
figure('NumberTitle', 'off', 'Name', 'P(Y=1|X)');

% Fastron RBF
img_rbf = zeros(size(X2)); % n x m; 
K_rbf = rbf([X2(:) Y2(:)], X_train, g);
img_rbf(:) = 1./(1 + exp(-K_rbf*a));
subplot(4,2,1), imshow(flip(img_rbf,1));
title("Fastron RBF kernel");

% Fastron RBF Calibrated
img_rbf_calibrated = zeros(size(X2)); % n x m; 
img_rbf_calibrated(:) = 1./(1 + exp(A_fastron.*K_rbf*a+ B_fastron));
subplot(4,2,2), imshow(flip(img_rbf_calibrated,1));
title("Fastron RBF Calibrated");

% LR;
img_lr = zeros(size(X2));
y = ones(size(X2(:)));
img_lr(:) = 1./(1 + exp(-(K_rbf*w_lr' + b_lr)));
subplot(4,2,3), imshow(flip(img_lr,1));
title("Logistic Regression");

% LR Calibrated;
img_lr_calibrated = zeros(size(X2));
y = ones(size(X2(:)));
img_lr_calibrated(:) = 1./(1 + exp(A_lr.*(K_rbf * w_lr' + b_lr) + B_lr));
subplot(4,2,4), imshow(flip(img_lr_calibrated,1));
title("Logistic Regression Calibrated");

% NN; 
img_mlp = zeros(size(X2));
F_mlp_test = feedforward([X2(:) Y2(:)], w_mlp, b_mlp);
img_mlp(:) = 1./ (1 + exp(-F_mlp_test));
subplot(4,2,5), imshow(flip(img_mlp,1));
title("Multi-layer Perceptron");

% NN Calibrated;
img_mlp_calibrated = zeros(size(X2));
img_mlp_calibrated(:) = 1./ (1 + exp(A_mlp.*(F_mlp_test)+B_mlp));
subplot(4,2,6), imshow(flip(img_mlp_calibrated,1));
title("Multi-layer Perceptron Calibrated");

% IVM;
img_ivm = zeros(size(X2));
K_ivm = rbf([X2(:) Y2(:)], S, g);
if useUnbiasedVersion
    F_ivm = K_ivm*a_ivm;
else
    F_ivm = K_ivm*a_ivm(1:end-1) + a_ivm(end);
end
img_ivm(:) = 1./ (1 + exp(-F_ivm));
subplot(4,2,7), imshow(flip(img_ivm,1));
title("IVM");

% IVM Calibrated
img_ivm_calibrated = zeros(size(X2));
img_ivm_calibrated(:) = 1./ (1 + exp(A_ivm * F_ivm + B_ivm));
subplot(4,2,8), imshow(flip(img_ivm_calibrated,1));
title("IVM Calibrated");

%% Bar graph for scores;
scoreFuns = {@nll, @brierScore};
names = ["Negative Log Loss", "Brier Score"];
for i=1:size(scoreFuns,2)
    figure('NumberTitle', 'off');
    score = zeros(4,2);
    p = [p_rbf(:) p_rbf_calibrated(:) p_lr_test(:) p_lr_calibrated(:) p_mlp(:) p_mlp_calibrated(:) p_ivm(:) p_ivm_calibrated(:)];
    scoreFun = scoreFuns{i};
    for j=1:size(p,2)/2
        score(j,1) = scoreFun(p(:, 2*j-1),y_test); % uncalibrated;
        score(j,2) = scoreFun(p(:, 2*j),y_test); % calibrated;
    end
    h = bar(score);
    set(gca,'xticklabel', ["Fastron" "LogReg" "MLP" "IVM"]); 
    l = cell(1,2);
    l{1}='Uncalibrated'; l{2}='Calibrated';  
    legend(h,l);
    title(names{i})
end

%% Table for precision, recall and confidence percentage
results = zeros(size(X_test,1),5,4); 
counts = [y_pred_fastron y_pred_lr_test y_pred_mlp y_pred_ivm y_pred_bagging_test ]; 

for i=1:size(results,2)
    results(:,i,1) = (counts(:,i) == 1) & (y_test == 1);
    results(:,i,2) = (counts(:,i) == 1) & (y_test ~= 1);
    results(:,i,3) = (counts(:,i) ~= 1) & (y_test ~= 1);
    results(:,i,4) = (counts(:,i) ~= 1) & (y_test == 1);
end
acc = sum(results(:,:,1) + results(:,:,3), 1)/size(y_test, 1);
tpr = sum(results(:,:,1),1)./(sum(results(:,:,1),1) + sum(results(:,:,2),1));
tnr = sum(results(:,:,3),1)./(sum(results(:,:,3),1) + sum(results(:,:,4),1));
recall_p = sum(results(:,:,1),1)./(sum(results(:,:,1),1) + sum(results(:,:,4),1));
recall_n = sum(results(:,:,3),1)./(sum(results(:,:,3),1) + sum(results(:,:,2),1));

conf = sum(p > 0.8 | p < .2, 1)/size(y_test,1);
conf_uncalibrated = [conf(:,1:2:7),0]; % TODO: Add Bagged Trees; 
conf_calibrated = [conf(:,2:2:8),0];

T = table(acc',tpr',recall_p',tnr',recall_n', conf_uncalibrated', conf_calibrated', ...
    'RowNames',["Fastron","LogReg","MLP","IVM","Bagged Trees"],...
    'VariableNames',{'accuracy', 'tpr','recall_pos','tnr', 'recall_neg', ...
    'high_confidence_uncalibrated', 'high_confidence_calibrated'
    });

% uitable('Data',T{:,:},'ColumnName', T.Properties.VariableNames,...
%     'RowName',T.Properties.RowNames);
% Get the table in string form.
TString = evalc('disp(T)');
% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');

% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');
% Output the table using the annotation command.
annotation(gcf,'Textbox','String',TString,'Interpreter','Tex',...
    'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);