close all;

% initialize matlab util
run('/home/nikhildas/workspace/matlab_util/init.m');

g = 0.2;

n = 100;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);
[X, Y] = meshgrid(x, y);
Z = X.^2 + Y.^2;
e = rand(n, n);
Z = Z + e; % add gaussian noise to the input;
shape = [n, n];

tic()
gprMdl = fitrgp([X(:) Y(:)], Z(:), 'KernelFunction', 'squaredexponential', ...
                'Sigma', sqrt(1.0/(2*g)));
t_train = toc()

tic();
[mu, std, yint] = predict(gprMdl, [X(:) Y(:)]);
t_test = toc()

fprintf("Training time for (%d, %d) training data: %f; ...Test time: %f", ...
    size(X(:), 1), 2, t_train, t_test);

%% Plot original data points and fitted surface;
% plot3(X, Y, Z, 'b.');
figure();
% surf(X, Y, Z, Z);
% colorbar;
% hold on;

plot3(X, Y, reshape(mu, shape), 'r.'); % plot mean;
hold on;
surf(X, Y, reshape(mu + std, shape));
alpha 0.1;
hold on;
surf(X, Y, reshape(mu - std, shape));
alpha 0.1;
% % shadedErrorBar(x, mu, 2*std);
% % legend({'data', 'prediction'});
title(sprintf('Gaussian Process with gamma:{%d}', g));
