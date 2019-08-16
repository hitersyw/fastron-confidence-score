dataset = "CSpace1";

% Hyper-parameters for Fastron learning;
iterMax = 5000;
g = 1.;
beta = 1.;
Smax = 2000;
num_bins = 10;

x_min = [-4, -4];
x_max = [4, 4];
r = 1;

% training_file = "./circle_training.csv";
% test_file = "./circle_test.csv";
training_file_spec = "./data/%s_training.csv";
test_file_spec = "./data/%s_test.csv";

training_file = sprintf(training_file_spec, dataset);
test_file = sprintf(test_file_spec, dataset);
[train_headers, X_train, y_train] = loadData(training_file);
[test_headers, X_test, y_test] = loadData(test_file);

% Plot test set;
figure('NumberTitle', 'off');
subplot(2,3,1);
x1 = X_test(y_test == 1, :);
x2 = X_test(y_test == -1, :);

scatter(x1(:, 1), x1(:, 2), 'r', 'filled');
hold on;
scatter(x2(:, 1), x2(:, 2), 'b', 'filled');
title("Original");

%% Fastron learning with Rational Quadratic Kernel
[a, F, K, iter]=trainFastron(X_train, y_train, @rq, iterMax, Smax, beta, g);
K_test_rq = rq(X_test, X_train(a~=0,:), g); % n x m; 
F_test_rq = K_test_rq*a(a~=0);
x_pos = X_test(F_test_rq >= 0, :);
x_neg = X_test(F_test_rq < 0, :);

subplot(2,3,2);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Fastron RQ");

%% Fastron learning with RBF; 
[a, F, K, iter]=trainFastron(X_train, y_train, @rbf, iterMax, Smax, beta, g);
K_test_rbf = rbf(X_test, X_train(a~=0,:), g); % n x m; 
F_test_rbf = K_test_rbf*a(a~=0);
x_pos = X_test(F_test_rbf >= 0, :);
x_neg = X_test(F_test_rbf < 0, :);

subplot(2,3,3);
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
data = values.test_output;
y_pred_lr = data(:,1);
p_lr = data(:, 3);

x_pos = X_test(y_pred_lr == 1, :);
x_neg = X_test(y_pred_lr == -1, :);

subplot(2,3,4);
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
x_neg = X_test(y_pred_mlp == -1, :);

subplot(2,3,5);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Multi-layer Perceptron");

%% Reliability diagrams; 
p_rq = 1./(1 + exp(-F_test_rq));
p_rbf = 1./(1 + exp(-F_test_rbf));
plotReliability([p_rq(:) p_rbf(:) p_lr(:) p_mlp(:)], y_test, num_bins, ...
      ["Fastron RQ" "Fastron RBF" "LogRegression" "MLP"]); 
% plotReliability([p_lr(:), p_mlp(:)], y_test, num_bins, ...
%     ["LogRegression" "MLP"]); 

%% Probability plots
[X2,Y2] = meshgrid(linspace(x_min(1), x_max(1),100), linspace(x_min(2),x_max(2),100));
figure('NumberTitle', 'off', 'Name', 'Probability grayscale');

% Fastron RQ
img_rq = zeros(size(X2)); % n x m; 
K_rq = rq([X2(:) Y2(:)], X_train(a~=0,:), g);
img_rq(:) = 1./(1 + exp(-K_rq*a(a~=0)));
subplot(2,2,1), imshow(flip(img_rq,1));
title("Fastron RQ kernel");

% Fastron RBF
img_rbf = zeros(size(X2)); % n x m; 
K_rbf = rbf([X2(:) Y2(:)], X_train(a~=0,:), g);
img_rbf(:) = 1./(1 + exp(-K_rbf*a(a~=0)));
subplot(2,2,2), imshow(flip(img_rbf,1));
title("Fastron RBF kernel");

% LR
img_lr = zeros(size(X2));
y = ones(size(X2(:)));
img_lr(:) = 1./(1 + exp(-y.*([X2(:) Y2(:)] * w_lr' + b_lr)));
subplot(2,2,3), imshow(flip(img_lr,1));
title("Logistic Regression");

% NN; 
img_mlp = zeros(size(X2));
h1 = ([X2(:) Y2(:)] * w_mlp{1} + b_mlp{1}');
h1(h1 < 0) = 0; % a1
h2 = h1 * w_mlp{2} + b_mlp{2}';
h2(h2 < 0) = 0; % a2
h3 = h2 * w_mlp{3} + b_mlp{3};
% TODO: Should the last layer be a sigmoid function?
% h3(h3 < 0) = 0; Relu or Sigmoid in the end?
h3 = 1./ (1 + exp(-y.*(h3)));
img_mlp(:) = h3;
subplot(2,2,4), imshow(flip(img_mlp,1));
title("Multi-layer Perceptron");