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

%% Plot the original data points; 
shape = [5, 15, 15]; % (theta, y, x);
x1 = reshape(X(:, 1), [5, 15, 15]);
x2 = reshape(X(:, 2), [5, 15, 15]);
theta = reshape(X(:, 4), [5, 15, 15]);
y = reshape(y, shape);

X_t = reshape(X, [5, 15, 15, 4]);

% fix theta;
for i = 1:5
    X_test = reshape(X_t(i, :, :, :), [], 4); % fix theta;
    [mu, std, yint] = predict(gprMdl, X_test);
    
    p1 = mu + std;
    p2 = mu - std;
    
    figure();
    
    % sigma plots; 
    subplot(2, 2, 1);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y(i, :, :),[15,15]), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold on;
    % prediction
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(p1,[15,15]), ...
        'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(p2,[15,15]), ...
    'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('Confidence Interval');
    
    % error plot;
    subplot(2, 2, 2);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(mu,[15,15]), ...
    'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    residual = reshape(y(i, :, :), [15,15]) - reshape(mu, [15, 15]);
    quiver3(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :), [15, 15]), ...
        reshape(mu, [15,15]), zeros(15,15), zeros(15,15), residual);    
    hold off; 
    title('Error bar');
    
    % MSE;
    subplot(2, 2, 3);
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), residual.^2, ...
        'FaceColor','b');
    title('MSE');
end
% surf(X(:,1), X(:, 2), p, p, 'b.');
% colorbar; 

