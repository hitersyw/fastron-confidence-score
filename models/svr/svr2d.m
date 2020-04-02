% Fit a SVR model on a 2d dataset. 
close all;
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
svrMdl = fitrsvm([X(:) Y(:)], Z(:), 'KernelFunction', 'rbf', ...
                'KernelScale','auto',...
                'Solver','SMO', ...
                'Standardize',false, ...
                'verbose',1 );
t_train = toc()

tic();
y_pred = predict(svrMdl, [X(:) Y(:)]);
t_test = toc()

fprintf("Training time for (%d, %d) training data: %f; ...Test time: %f", ...
    size(X(:), 1), 2, t_train, t_test);
epsilon = svrMdl.ModelParameters.Epsilon;

%% Plot original data points and fitted surface;
figure();

plot3(X, Y, Z, 'r.'); % plot mean;
hold on;
surf(X, Y, reshape(y_pred + epsilon, shape));
alpha 0.1;
hold on;
surf(X, Y, reshape(y_pred - epsilon, shape));
alpha 0.1;
title('Support Vector Regression for 2d');
