clear; close all;
rng(0);

%% Hyper-parameters;
g = 20;

dataset = 'collision_score';

%% Load collision_score data
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
score_dict = load(sprintf('/home/jamesdi1993/workspace/arclab/fastron_experimental/fastron_vrep/constraint_analysis/log/%s_n1125.mat', dataset));
score = getfield(score_dict, dataset);
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

%% Fit Gaussian Regression on the data;
% gprMdl = fitrgp(X, y, 'KernelFunction', 'squaredexponential', ...
%     'Sigma', sqrt(1.0/(2*g)));

gprMdl = fitrgp(X, y,'KernelFunction','ardsquaredexponential','FitMethod','exact','PredictMethod','exact',...
            'Basis','Constant','Optimizer','quasinewton', ...
            'Sigma', sqrt(1.0/(2*g)),'Standardize',false, 'verbose',1);

%% Plot the fitted gaussian curves; 
shape = [5, 15, 15]; % (theta, y, x);
x1 = reshape(X(:, 1), [5, 15, 15]);
x2 = reshape(X(:, 2), [5, 15, 15]);
theta = reshape(X(:, 4), [5, 15, 15]);

y_t = y;
y_t = reshape(y_t, shape);

X_t = reshape(X, [5, 15, 15, 4]);

% fix theta;
for i = 1:5
    X_test = reshape(X_t(i, :, :, :), [], 4); % fix theta;
    [mu, std, yint] = predict(gprMdl, X_test);
    
%     p1 = mu + std;
%     p2 = mu - std;
    p1 = yint(:, 1);
    p2 = yint(:, 2);
    
    figure();
    
    % sigma plots; 
    subplot(2, 2, 1);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y_t(i, :, :),[15,15]), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold on;
    % prediction
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(p1,[15,15]), ...
        'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(p2,[15,15]), ...
        'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('95% Confidence Interval');
    
    % error plot;
    subplot(2, 2, 2);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(mu,[15,15]), ...
    'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    residual = reshape(y_t(i, :, :), [15,15]) - reshape(mu, [15, 15]);
    quiver3(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :), [15, 15]), ...
        reshape(mu, [15,15]), zeros(15,15), zeros(15,15), residual);
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    hold off; 
    title('Error bar');
    
    % MSE;
    subplot(2, 2, 3);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), residual.^2, ...
        'FaceColor','b', 'EdgeColor','none');
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('MSE');
    
    sgtitle(sprintf("%s for theta: %.2f", strrep(dataset, "_", " "), theta(i, 1, 1)));
end

% Compute MSE Loss;
[mu, std, yint] = predict(gprMdl, X);
mu = max(min(mu, 1),0); % clip the value between 0 and 1;
eps = mu - y;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);
% surf(X(:,1), X(:, 2), p, p, 'b.');
% colorbar; 

