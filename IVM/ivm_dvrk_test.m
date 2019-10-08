clear; close all;
rng(0)

%% Load dataset
dir = "./data";
sample_file_spec = dir + "/joint_angle_sample_X_n%d_arm%d.csv";
collision_label_spec = dir + "/collision_state_y_n%d_arm%d.csv";
arm = 1;
n = 10000;

sample_file = sprintf(sample_file_spec, n, arm);
collision_label_file = sprintf(collision_label_spec, n, arm);
X = dlmread(sample_file,',',0,0);
y = dlmread(collision_label_file,',',0,0);
y = (y+1)/2; % input is 1 and -1; convert to 1 and 0;

training_p = 0.8;
validation_p = 0.1;
test_p = 0.1;

% train, validation and test splits;
% X_train = X(1:training_p * n, :);
% y_train = y(1:training_p * n, :);
X_train = X(1:2, :);
y_train = y(1:2, :);
X_holdout = X(training_p*n+1:(validation_p + training_p)*n, :);
y_holdout = y(training_p*n+1:(validation_p + training_p)*n, :);
X_test = X((validation_p + training_p)*n+1:n, :);
y_test = y((validation_p + training_p)*n+1:n, :);

%% IVM
lambda = 5;
g = 20;
useUnbiasedVersion = false;

profile on;
K = rbf(X_train, X_train, g);
if useUnbiasedVersion
    [a_ivm, S, idx] = ivmTrain(X_train, y_train, K, lambda);
else
    [a_ivm, S, idx] = ivmTrain2(X_train, y_train, K, lambda);
end
profile off;
% profsave(profile('info'), './log/42035');

if useUnbiasedVersion
    F_test_IVM = rbf(X_test, S, g)*a_ivm;
else
    F_test_IVM = rbf(X_test, S, g)*a_ivm(1:end-1) + a_ivm(end);
end
p_ivm = 1./(1 + exp(-F_test_IVM));

y_pred_ivm = (sign(F_test_IVM)+1)/2;
acc = sum(y_pred_ivm == y_test) / size(y_test, 1);
fprintf("The accuracy of IVM is: %s", acc);
