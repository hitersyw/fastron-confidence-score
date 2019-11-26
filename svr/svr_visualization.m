clear; close all;
rng(0);
init;

%% Hyper-parameters;
g = 20;
numIter = 5;
dataset = 'reachability_score';

%% Load collision_score data
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
score_dict = load(sprintf(base_dir + "log/%s_n1125.mat", dataset));
score = getfield(score_dict, dataset);
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

%% Fit Gaussian Regression on the data;
% gprMdl = fitrgp(X, y, 'KernelFunction', 'squaredexponential', ...
%     'Sigma', sqrt(1.0/(2*g)));

svrMdl = fitrsvm(X, y,'KernelFunction','rbf', 'KernelScale','auto',...
            'Solver','SMO', ...
            'Standardize',false, 'verbose',1 ,'IterationLimit', numIter);
t = numIter;

%% Plot the fitted SVR surfaces;
figure(1);
shape = [5, 15, 15]; % (theta, y, x);
x1 = reshape(X(:, 1), [5, 15, 15]);
x2 = reshape(X(:, 2), [5, 15, 15]);
theta = reshape(X(:, 4), [5, 15, 15]);

y_t = y;
y_t = reshape(y_t, shape);

X_t = reshape(X, [5, 15, 15, 4]);

while ~svrMdl.ConvergenceInfo.Converged
    % Take the first plot; 
    figure(1); clf
    i = 1;
    X_test = reshape(X_t(i, :, :, :), [], 4); % fix theta;
    y_pred = predict(svrMdl, X_test);

        % Prediction and Original data;
    subplot(2, 2, 1);
    h1 = surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y_t(i, :, :),[15,15]), ...
        'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    
    
    % prediction
    h2 = surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y_pred, [15,15]), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    title('Prediction and Original Data');
    
    % error plot;
    subplot(2, 2, 2);
    h3 = surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y_pred,[15,15]), ...
    'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    residual = reshape(y_t(i, :, :), [15,15]) - reshape(y_pred, [15, 15]);
    quiver3(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :), [15, 15]), ...
        reshape(y_pred, [15,15]), zeros(15,15), zeros(15,15), residual);
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    hold off; 
    title('Error bar');
    
    % Sqaured Error;
    subplot(2, 2, 3);
    h4 = surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), residual.^2, ...
        'FaceColor','b', 'EdgeColor','none');
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('Squared Error');
    
    % Legend
    lh = legend([h1,h2,h4], {'Original', 'Predicted', 'Error'}, 'location','southeast');
    set(lh,'position',[.7 .3 .1 .1])
    sgtitle(sprintf("%s for theta: %.2f; Iteration:%d", strrep(dataset, "_", " "), theta(i, 1, 1), t));
    
    % resume training
    % pause(0.1);
    svrMdl = resume(svrMdl, numIter);
    t = t + numIter;
end

% Compute MSE Loss;
y_pred = predict(svrMdl, X);
mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
eps = y_pred - y;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);
% surf(X(:,1), X(:, 2), p, p, 'b.');
% colorbar; 

