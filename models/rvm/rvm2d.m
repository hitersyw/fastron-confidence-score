close all;
init;

g = 0.2;
n = 30;

x = linspace(-2, 2, n);
y = linspace(-2, 2, n);
[xx, yy] = meshgrid(x, y);
zz = xx.^2 + yy.^2;
e = rand(n, n);
zz = zz + e; % add gaussian noise to the input;
shape = [n, n];

X = xx(:);
Y = yy(:);
Z = zz(:); 
BASIS = rbf([X(:) Y(:)], [X(:) Y(:)], g);
M = size(BASIS,2);

OPTIONS		= SB2_UserOptions('iterations',maxIter,...
							  'diagnosticLevel', 2,...
							  'monitor', 10);
SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

tic();
[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
    SparseBayes('Gaussian', BASIS, Z, OPTIONS, SETTINGS);
t_train = toc();

w_infer						= zeros(M,1);
w_infer(PARAMETER.Relevant)	= PARAMETER.Value;

tic();
y_pred = BASIS*w_infer;
t_test = toc();

fprintf("Training time for (%d, %d) training data: %f; Test time: %f", ...
    size(X, 1), 2, t_train, t_test);

%% Plot original data points and fitted surface;
% plot3(X, Y, Z, 'b.');
figure();
% surf(X, Y, Z, Z);
% colorbar;
% hold on;
S = find(w_infer);
R = find(~w_infer);


plot3(X(R, :), Y(R, :), Z(R), 'b.', 'MarkerSize',3); % plot unrelevant;
hold on;
plot3(X(S, :), Y(S, :), Z(S), 'r.', 'MarkerSize',10); % plot support points;

hold on;
surf(xx, yy, reshape(y_pred,shape));
alpha 0.1;
% % shadedErrorBar(x, mu, 2*std);
% % legend({'data', 'prediction'});
title('Relevance Vector Machine for 2d');
