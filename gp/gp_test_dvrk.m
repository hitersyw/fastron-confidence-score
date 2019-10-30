clear; close all;
rng(0);

%% Hyper-parameters;
g = 20;

%% load collision_score data
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
score_dict = load('/home/nikhildas/workspace/fastron_experimental/fastron_vrep/constraint_analysis/log/collision_score_n1125.mat');
score = getfield(score_dict, 'collision_score');
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";
spec = "%2f";

%% Fit Gaussian Regression on the data;
gprMdl = fitrgp(X, y, 'KernelFunction', 'squaredexponential', ...
    'Sigma', sqrt(1.0/(2*g)));
p = min(max(resubPredict(gprMdl),0), 1); % clip the values;

%% Plot the original data points; 
figure();
plot3(X(:,1), X(:,2), y, 'b.');
xlabel('X');
ylabel('Y');
zlabel('score');
legend({'data', 'prediction'});

%% Divide the data into grids;
figure();
shape = [15,15,5];
x = reshape(X(:, 1), shape); 
y = reshape(X(:, 2), shape);
theta = reshape(X(:, 4), shape);
score = reshape(p, shape);
surf(x(:, :, 1), y(:, :, 1), score(:, :, 1), score(:, :, 1)); % use only 1 theta value;
xlabel('X');
ylabel('Y');
zlabel('score');
% figure();
% surf(X(:,1), X(:, 2), p, p, 'b.');
% colorbar; 

