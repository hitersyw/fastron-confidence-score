close all;

% initialize matlab util
run('/home/jamesdi1993/workspace/matlabCollection/init.m');

g = 20;

figure();
n = 1000;
x = linspace(-2, 2, n);
y = sin(2*pi*x);
e = rand(1, n);
z = y + e;
plot(x, z, 'b.');
hold on;


svrMdl = fitrsvm(x', z', 'KernelFunction', 'rbf', ...
                'KernelScale','auto',...
                'Solver','SMO', ...
                'Standardize',false, ...
                'verbose',1 );
epsilon = svrMdl.ModelParameters.Epsilon;
y_pred = predict(svrMdl, x');
plot(x, y_pred, 'r');
shadedErrorBar(x, y_pred, epsilon.*ones(numel(y_pred), 1));
legend({'data', 'prediction'});
title('1d Support Vector Regression');
