% Script used for testing Platt Scaling;
dataset = "CSpace1";

% Hyper-parameters for Fastron-Learning;
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
v_ratio = 0.3; % usual choice of hold-out set;

training_file_spec = "./data/%s_training.csv";
validation_file_spec = "./data/%s_validation.csv";
test_file_spec = "./data/%s_test.csv";

training_file = sprintf(training_file_spec, dataset);
validation_file = sprintf(validation_file_spec, dataset);
test_file = sprintf(test_file_spec, dataset);
[X_train, y_train] = loadData2(training_file);
[X_holdout, y_holdout] = loadData2(validation_file);
[X_test, y_test] = loadData2(test_file);

% train Fastron w/ RBF kernel;
[a, F, K, iter]=trainFastron(X_train, y_train, @rbf, iterMax, Smax, beta, g);
K_test_rbf = rbf(X_test, X_train(a~=0,:), g); % n x m; 
F_test_rbf = K_test_rbf*a(a~=0);
p_rbf = 1./(1 + exp(-F_test_rbf)); % without calibration;

K_holdout = rbf(X_holdout, X_train(a~=0,:), g); % n x m; 
F_holdout = K_holdout*a(a~=0);
[A, B] = trainPlattScaling(F_holdout, y_holdout, iterMax, eps, lr);

p_rbf_calibrated = 1./(1 + exp(A.*F_test_rbf + B));

% multi_layer perceptron;
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

plotReliability([p_rbf p_rbf_calibrated], y_test, num_bins,... 
["Original Fastron", "Fastron RBF Calibrated"]); 

% Plot probability curves;
[X2,Y2] = meshgrid(linspace(x_min(1), x_max(1),100), linspace(x_min(2),x_max(2),100));
figure('NumberTitle', 'off', 'Name', 'Probability grayscale');

img_rbf = zeros(size(X2)); % n x m; 
K_rbf = rbf([X2(:) Y2(:)], X_train(a~=0,:), g);
img_rbf(:) = 1./(1 + exp(-K_rbf*a(a~=0)));
subplot(2,1,1), imshow(flip(img_rbf,1));
title("Fastron RBF kernel Original");

img_rbf_calibrated = zeros(size(X2)); % n x m; 
img_rbf_calibrated(:) = 1./(1 + exp(A.*K_rbf*a(a~=0)+ B));
subplot(2,1,2), imshow(flip(img_rbf_calibrated,1));
title("Fastron RBF kernel Calibrated");

