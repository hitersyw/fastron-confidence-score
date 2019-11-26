clear; close all;
rng(0);
init;

%% Hyper-parameters;
dataset = 'collision_score';

%% Load collision_score data
score_dict = load(sprintf(base_dir + "log/%s_n1125.mat", dataset));
score = getfield(score_dict, dataset);
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

%% Fit Support Vector Regression on the data;
svrMdl = fitrsvm(X, y,'KernelFunction','rbf', 'KernelScale','auto',...
            'Solver','SMO', ...
            'Standardize',false, ...
            'Epsilon', 0.01, ...
            'verbose',1);
epsilon = svrMdl.ModelParameters.Epsilon;


%% Plot the fitted SVR surfaces; 
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
    y_test = y_t(i, :, :);
    y_pred = predict(svrMdl, X_test);
    
    xx = x1(i, :, :);
    yy = x2(i, :, :);
%     p1 = mu + std;
%     p2 = mu - std;

    figure();
    
    % Prediction and Original data;
    ha = tight_subplot(2, 2, [.1 .1],[.1 .2],[.1 .1]);
    
    axes(ha(1));
    h1 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_t(i, :, :),[15,15]), ...
        'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    
    
    % prediction
    h2 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred, [15,15]), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    title('Prediction and Original Data');
    
    % error plot;
    axes(ha(2));
    h3 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred,[15,15]), ...
    'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','none');
    hold on;
    residual = reshape(y_t(i, :, :), [15,15]) - reshape(y_pred, [15, 15]);
    quiver3(reshape(xx, [15,15]), reshape(yy, [15, 15]), ...
        reshape(y_pred, [15,15]), zeros(15,15), zeros(15,15), residual);
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    
    hold off; 
    title('Error bar');
    
    % Support Point Visualization
    axes(ha(3));
    xx = x1(i, :, :);
    yy = x2(i, :, :);
    plot3(xx(:), yy(:), y_test(:),'r.');
    hold on;
    % prediction
    h4 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred + epsilon,[15,15]), ...
        'FaceColor','y', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold on;
    surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred - epsilon,[15,15]), ...
        'FaceColor','y', 'FaceAlpha',1.0, 'EdgeColor','none');
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('Epsilon band');
    
    % Sqaured Error;
    axes(ha(4));
    h5 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), residual.^2, ...
        'FaceColor','b', 'EdgeColor','none');
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title('Squared Error');
    
    
    % Legend
    lh = legend([h1,h2,h4,h5], {'Original', 'Predicted', 'Support Boundary', 'Error'}, 'location','northeast');
    set(lh,'position',[.85 .85 .1 .1])
    sgtitle(sprintf("%s for theta: %.2f", strrep(dataset, "_", " "), theta(i, 1, 1)));
end

% Compute MSE Loss;
y_pred = predict(svrMdl, X);
mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
eps = y_pred - y;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);

