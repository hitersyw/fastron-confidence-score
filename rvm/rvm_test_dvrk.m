clear; close all;
rng('default');
init;

%% Hyper-parameters
g = 40;

%% Load collision_score data
dataset = 'reachability_score';
score_dict = load(sprintf(base_dir + "log/%s_n1125.mat", dataset));
score = getfield(score_dict, dataset);
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";

BASIS = rbf(X, X, g);
M = size(BASIS,2);


%% Define model parameters
maxIter = 500;
OPTIONS = SB2_UserOptions('iterations', maxIter,...
                          'diagnosticLevel', 2,...
						  'monitor', 10);
SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
    SparseBayes('Gaussian', BASIS, y, OPTIONS, SETTINGS);
w_infer						= zeros(M,1);
w_infer(PARAMETER.Relevant)	= PARAMETER.Value;

%% Plot the the fitted RVM surface;
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
    
    basis_test = rbf(X_test, X, g);
    
    y_pred = basis_test * w_infer;
    
    xx = x1(i, :, :);
    yy = x2(i, :, :);
%     p1 = mu + std;
%     p2 = mu - std;

    figure(1); clf;
    fig.InvertHardcopy = 'off'
    whitebg(1, 'k'); % set gray background;
    
    % Prediction and Original data;
    ha = tight_subplot(2, 2, [.1 .1],[.1 .2],[.1 .1]);
    
    axes(ha(1));
    h1 = plot3(xx(:), yy(:), y_t(i,:), 'w.');
    hold on;
    
    % prediction
    h2 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred, [15,15]), ...
        'FaceColor','r', 'EdgeColor', 'k', 'FaceAlpha', 1.0);
    hold off; 
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    grid off;
    title('Prediction and Original Data');
    
    % error plot;
    axes(ha(2));
    h3 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred,[15,15]), ...
    'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','k');
    hold on;
    residual = reshape(y_t(i, :, :), [15,15]) - reshape(y_pred, [15,15]);
    quiver3(reshape(xx, [15,15]), reshape(yy, [15, 15]), ...
        reshape(y_pred, [15,15]), zeros(15,15), zeros(15,15), residual, 0, 'Color', 'w'); % do not scale S;
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    grid off; hold off; 
    title('Error bar');
    
    % Support Point Visualization
    
    w_t = reshape(w_infer, shape);
    S = find(w_t(i, :, :));
    R = find(~w_t(i, :, :));
    axes(ha(3));
    xx = x1(i, :, :);
    yy = x2(i, :, :);
%     plot3(xx(:), yy(:), y_test(:),'w.'); % the rest of the points
%     hold on
    plot3(xx(S), yy(S), y_test(S), 'g.', 'markerSize', 10); % support points
    hold on; 
    % prediction
    h4 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), reshape(y_pred,[15,15]), ...
        'FaceColor','r', 'FaceAlpha',1.0, 'EdgeColor','k');
    hold on;
    xlabel('X')
    ylabel('Y');
    zlabel('score');
    title('Relevance points');
    
    % Squared Error;
    axes(ha(4));
    h5 = surf(reshape(xx, [15,15]), reshape(yy,[15,15]), residual.^2, ...
        'FaceColor','b', 'EdgeColor','k');
    xlabel('X');
    ylabel('Y');
    zlabel('Error');
    grid off;
    title('Squared Error');
    
    % Legend
    lh = legend([h1,h2,h5], {'Original', 'Predicted', 'Error'}, 'location','northeast');
    set(lh,'position',[.85 .85 .1 .1])
    sgtitle(sprintf("%s for theta: %.2f", strrep(dataset, "_", " "), theta(i, 1, 1)));
end

% Compute MSE Loss;
y_pred = BASIS * w_infer;
mu = max(min(y_pred, 1),0); % clip the value between 0 and 1;
eps = y_pred - y;
l = eps' * eps / n;
fprintf("MSE Loss: %.4f\n", l);