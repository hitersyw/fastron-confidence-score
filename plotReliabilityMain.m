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

training_file = sprintf(training_file_spec, dataset);
validation_file = sprintf(validation_file_spec, dataset);
test_file = sprintf(test_file_spec, dataset);
[X_train, y_train] = loadData2(training_file);
[X_holdout, y_holdout] = loadData2(validation_file);
[X_test, y_test] = loadData2(test_file);

%% Plot test set;
figure('NumberTitle', 'off');
subplot(2,2,1);
x1 = X_test(y_test == 1, :);
x2 = X_test(y_test == -1, :);

scatter(x1(:, 1), x1(:, 2), 'r', 'filled');
hold on;
scatter(x2(:, 1), x2(:, 2), 'b', 'filled');
title("Original");

%% Fastron learning with RBF; 
[a, F, K, iter]=trainFastron(X_train, y_train, @rbf, iterMax, Smax, beta, g);
K_test_rbf = rbf(X_test, X_train, g); % n x m; 
F_test_rbf = K_test_rbf*a;
x_pos = X_test(F_test_rbf >= 0, :);
x_neg = X_test(F_test_rbf < 0, :);
p_rbf = 1./(1 + exp(-F_test_rbf)); % without calibration;

subplot(2,2,2);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Fastron RBF");

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

subplot(2,2,3);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Logistic Regression");

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

subplot(2,2,4);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Multi-layer Perceptron");

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
h1 = X_holdout * w_mlp{1} + b_mlp{1}';
h1(h1 < 0) = 0; % a1
h2 = h1 * w_mlp{2} + b_mlp{2}';
h2(h2 < 0) = 0; % a2
F_holdout_mlp = h2 * w_mlp{3} + b_mlp{3}';
[A_mlp, B_mlp] = trainPlattScaling(F_holdout_mlp, y_holdout, iterMax, eps, lr);


% TODO: Refactor feedfoward into a separate function;
h1 = X_test * w_mlp{1} + b_mlp{1}';
h1(h1 < 0) = 0; % a1
h2 = h1 * w_mlp{2} + b_mlp{2}';
h2(h2 < 0) = 0; % a2
F_test_mlp = h2 * w_mlp{3} + b_mlp{3}';
p_mlp_calibrated = 1./(1 + exp(A_mlp.* F_test_mlp + B_mlp));

%% Reliability diagrams; 
plotReliability([p_rbf(:) p_rbf_calibrated(:) p_lr_test(:) p_lr_calibrated(:) p_mlp(:) p_mlp_calibrated(:)],...
    y_test, num_bins, ["Fastron RBF" "Fastron RBF Calibrated" "LogRegression" "LogRegression Calibrated" "MLP" "MLP Calibrated"]); 
% plotReliability([p_lr(:), p_mlp(:)], y_test, num_bins, ...
%     ["LogRegression" "MLP"]); 

%% Probability plots
[X2,Y2] = meshgrid(linspace(x_min(1), x_max(1),100), linspace(x_min(2),x_max(2),100));
figure('NumberTitle', 'off', 'Name', 'Probability grayscale');

% Fastron RBF
img_rbf = zeros(size(X2)); % n x m; 
K_rbf = rbf([X2(:) Y2(:)], X_train, g);
img_rbf(:) = 1./(1 + exp(-K_rbf*a));
subplot(3,2,1), imshow(flip(img_rbf,1));
title("Fastron RBF kernel");

% Fastron RBF Calibrated
img_rbf_calibrated = zeros(size(X2)); % n x m; 
img_rbf_calibrated(:) = 1./(1 + exp(A_fastron.*K_rbf*a+ B_fastron));
subplot(3,2,2), imshow(flip(img_rbf_calibrated,1));
title("Fastron RBF Calibrated");

% LR;
img_lr = zeros(size(X2));
y = ones(size(X2(:)));
img_lr(:) = 1./(1 + exp(-(K_rbf*w_lr' + b_lr)));
subplot(3,2,3), imshow(flip(img_lr,1));
title("Logistic Regression");

% LR Calibrated;
img_lr_calibrated = zeros(size(X2));
y = ones(size(X2(:)));
img_lr_calibrated(:) = 1./(1 + exp(A_lr.*(K_rbf * w_lr' + b_lr) + B_lr));
subplot(3,2,4), imshow(flip(img_lr_calibrated,1));
title("Logistic Regression Calibrated");

% NN; 
img_mlp = zeros(size(X2));
h1 = ([X2(:) Y2(:)] * w_mlp{1} + b_mlp{1}');
h1(h1 < 0) = 0; % a1
h2 = h1 * w_mlp{2} + b_mlp{2}';
h2(h2 < 0) = 0; % a2
h3 = h2 * w_mlp{3} + b_mlp{3}';
h3 = 1./ (1 + exp(-h3));
img_mlp(:) = h3;
subplot(3,2,5), imshow(flip(img_mlp,1));
title("Multi-layer Perceptron");

% NN Calibrated;
img_mlp_calibrated = zeros(size(X2));
h1 = ([X2(:) Y2(:)] * w_mlp{1} + b_mlp{1}');
h1(h1 < 0) = 0; % a1
h2 = h1 * w_mlp{2} + b_mlp{2}';
h2(h2 < 0) = 0; % a2
h3 = h2 * w_mlp{3} + b_mlp{3}';
img_mlp_calibrated(:) = 1./ (1 + exp(A_mlp.*(h3)+B_mlp));
subplot(3,2,6), imshow(flip(img_mlp_calibrated,1));
title("Multi-layer Perceptron Calibrated");