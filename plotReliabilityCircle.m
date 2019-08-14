% Hyper-parameters for Fastron learning;
iterMax = 5000;
g = 1.;
beta = 1.;
Smax = 2000;
num_bins = 10;

x_min = [-2, -2];
x_max = [2, 2];
r = 1;

% Generate unit-circle dataset;
% r = 1; 
% X = x_min + (x_max - x_min).*rand(total_points, 2);

% % training set;
% X_train = X(1:(1-total_test_p)*total_points, :);
% dist_train = pdist2([0 0], X_train).';
% x1 = X_train(dist_train <= 1, :);
% x2 = X_train(dist_train > 1, :);
% X_train = cat(1, x1, x2);
% y_train = cat(1, ones(length(x1), 1), -1.*ones(length(x2), 1));
% 
% % test set;
% X_test = X((1-total_test_p)*total_points+1:end, :);
% dist_test = pdist2([0 0], X_test).';
% x3 = X_test(dist_test <= 1, :);
% x4 = X_test(dist_test > 1, :);
% X_test = cat(1, x3, x4);
% y_test = cat(1, ones(length(x3), 1), -1.*ones(length(x4), 1));

training_file = "./circle_training.csv";
test_file = "./circle_test.csv";
[train_headers, X_train, y_train] = loadData(training_file);
[test_headers, X_test, y_test] = loadData(test_file);

% Plot test set;
figure('NumberTitle', 'off');
subplot(1,3,1);
dist_test = pdist2([0 0], X_test).';
x1 = X_test(dist_test <= r, :);
x2 = X_test(dist_test > r, :);

scatter(x1(:, 1), x1(:, 2), 'r', 'filled');
hold on;
scatter(x2(:, 1), x2(:, 2), 'b', 'filled');
title("Original");

% Fastron learning;
[a, F, K, iter]=trainFastron(X_train, y_train, iterMax, Smax, beta, g);
r2 = 1+g/2*pdist2(X_test, X_train(a~=0,:)).^2; % n x m; 
K_test = 1./(r2.*r2);
F_test = K_test*a(a~=0);
x_pos = X_test(F_test >= 0, :);
x_neg = X_test(F_test < 0, :);

subplot(1,3,2);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Fastron with FK kernel");

% Load output from LR
log_reg_file="./logRegression_circle.csv";
fid = fopen(log_reg_file);
header = fgetl(fid);
data = dlmread(log_reg_file,',',0,0);
y_pred_lr = data(:, 1);
% TODO: Use header to find the data column;
p_lr = data(:, 3);

x_pos = X_test(y_pred_lr >= 0, :);
x_neg = X_test(y_pred_lr < 0, :);

subplot(1,3,3);
scatter(x_pos(:, 1), x_pos(:,2), 'r', 'filled');
hold on;
scatter(x_neg(:, 1), x_neg(:,2), 'b', 'filled');
title("Logistic Regression");

% Reliability diagram; 
p_fastron = 1./(1 + exp(-F_test));
plotReliability([p_fastron(:), p_lr(:)], y_test, num_bins, ["Fastron" "LogRegression"]); 

% Probability plots
[X2,Y2] = meshgrid(linspace(x_min(1), x_max(1),100), linspace(x_min(2),x_max(2),100));
figure('NumberTitle', 'off', 'Name', 'Probability grayscale');

% Kernel perceptron prediction;
img_fastron = zeros(size(X2));
r2 = 1+g/2*pdist2([X2(:) Y2(:)], X_train(a~=0,:)).^2; % n x m; 
K = 1./(r2.*r2);
img_fastron(:) = 1./(1 + exp(-K*a(a~=0)));
% img(:) = sign(exp(-gamma*(pdist2([X2(:) Y2(:)], X(weights~=0,:), @modDistance).^2))*weights(weights~=0));
% imagesc([x_min(1), x_max(1)], [x_min(2), x_max(2)], img)
subplot(1,1,1), imshow(img_fastron);

% Prediction rule for LR
img_lr = zeros(size(X2));
r2 = 1+g/2*pdist2([X2(:) Y2(:)], X_train(a~=0,:)).^2; % n x m; 
title("Fastron with FK Kernel");