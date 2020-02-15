clear; close all;
rng(0);
init;

%% Hyper-parameters;
box_constraint = 0.46416; %23;
kernel_scale = 0.46416; %0.3;
epsilon = 0.018726; %0.0002;
numIter = 1000;
maxIter = 10000; 
dataset = 'self_collision_score'; % 'reachability_score';

%% Load collision_score data
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
score_dict = load(sprintf(base_dir + "log/%s_n1053.mat", dataset));
score = getfield(score_dict, dataset);
X = score(:, [1,2,4]);

%%  Normalize the dataset;
xmax = max(X);
xmin = min(X);
scale_input = @(x) 2*(x - xmin)./(xmax - xmin) - 1; % Normalize input between -1 and 1;
X = scale_input(X);

% Fit the score; 
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

%% Fit Gaussian Regression on the data;
% gprMdl = fitrgp(X, y, 'KernelFunction', 'squaredexponential', ...
%     'Sigma', sqrt(1.0/(2*g)));
% find_weights = @(x) 1.0 / (1 - clip(x, 0.01));
find_weights = @(x) ones(numel(x), 1);
observation_weights = find_weights(y); 

svrMdl = fitrsvm(X,y,'KernelFunction','rbf',...
            'KernelScale', kernel_scale, ...
            'BoxConstraint', box_constraint, ...
            'Epsilon', epsilon, ...
            'Weights', observation_weights, ...
            'IterationLimit', numIter);

% svrMdl = fitrsvm(X,y,'KernelFunction','rbf',...
%          'OptimizeHyperparameters','auto',...
%          'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%          'expected-improvement-plus', ...
%          'Optimizer', 'gridsearch'));
       
t = numIter;

%% Plot the fitted SVR surfaces;
figure(1);
shape = [13, 9, 9]; % (theta, y, x);
xy_shape = shape(2:3);
x1 = reshape(X(:, 1), shape);
x2 = reshape(X(:, 2), shape);
theta = reshape(X(:, 3), shape);

y_t = y;
y_t = reshape(y_t, shape);

X_t = reshape(X, [shape, 3]);

while ~svrMdl.ConvergenceInfo.Converged & t < maxIter    
    % resume training
    % pause(0.1);
    svrMdl = resume(svrMdl, numIter);
    t = t + numIter;
    
    
    
    % Take the first plot; 
    figure(1); clf
    i = 1;
    X_test = reshape(X_t(i, :, :, :), [], 3); % fix theta;
    y_pred = predict(svrMdl, X_test);

    % Prediction and Original data;
    subplot(2, 2, 1);
    h1 = surf(reshape(x1(i, :, :), xy_shape), reshape(x2(i, :, :),xy_shape), reshape(y_t(i, :, :),xy_shape), ...
        'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    
    
    % prediction
    h2 = surf(reshape(x1(i, :, :), xy_shape), reshape(x2(i, :, :), xy_shape), reshape(y_pred, xy_shape), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    title('Prediction and Original Data');
    
    % error plot;
    subplot(2, 2, 2);
    h3 = surf(reshape(x1(i, :, :), xy_shape), reshape(x2(i, :, :), xy_shape), reshape(y_pred, xy_shape), ...
    'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    residual = reshape(y_t(i, :, :), xy_shape) - reshape(y_pred, xy_shape);
    quiver3(reshape(x1(i, :, :), xy_shape), reshape(x2(i, :, :), xy_shape), ...
        reshape(y_pred, xy_shape), zeros(xy_shape), zeros(xy_shape), residual);
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    hold off; 
    title('Error bar');
    
    % Absolute Error;
    subplot(2, 2, 3);
    h4 = surf(reshape(x1(i, :, :), xy_shape), reshape(x2(i, :, :),xy_shape), abs(residual), ...
        'FaceColor','b', 'EdgeColor','none');
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('Absolute Error');
    
    % Legend
    lh = legend([h1,h2,h4], {'Original', 'Predicted', 'Error'}, 'location','southeast');
    set(lh,'position',[.7 .3 .1 .1])
    sgtitle(sprintf("%s for theta: %.2f; Iteration:%d", strrep(dataset, "_", " "), theta(i, 1, 1), t));
end

% Compute MSE Loss;
y_pred = predict(svrMdl, X);
mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
eps = y_pred - y;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);
fprintf("Maximum Loss: %.4f\n", max(abs(eps)));
% surf(X(:,1), X(:, 2), p, p, 'b.');
% colorbar; 

