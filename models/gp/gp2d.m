% Fit Gaussian Process on a 2d toy dataset. 
close all; clear;
init; 

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
t_train = toc();

tic();
[mu, std, yint] = predict(gprMdl, [X(:) Y(:)]);
t_test = toc();

fprintf("Training time for (%d, %d) training data: %f; ...Test time: %f", ...
    size(X(:), 1), 2, t_train, t_test);

%% Plot original data points and fitted surface;
figure();

plot3(X, Y, reshape(mu, shape), 'r.'); % plot mean;
hold on;
surf(X, Y, reshape(mu + std, shape));
alpha 0.1;
hold on;
surf(X, Y, reshape(mu - std, shape));
alpha 0.1;
title(sprintf('Gaussian Process with gamma:{%.2f}', g));
