close all;

g = 20;

% Set up dataset; 
figure();
n = 1000;
X = linspace(-2, 2, n);
y = sin(2*pi*X);
e = rand(1, n);
z = y + e;
plot(X, z, 'b.');
hold on;

% Compute kernel matrix; 
BASIS = rbf(X', X', g);
M = size(BASIS,2);

maxIter = 100;

% Set up RVM model
OPTIONS		= SB2_UserOptions('iterations',maxIter,...
							  'diagnosticLevel', 2,...
							  'monitor', 10);
SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
    SparseBayes('Gaussian', BASIS, z', OPTIONS, SETTINGS);

% Infered weights;
w_infer						= zeros(M,1);
w_infer(PARAMETER.Relevant)	= PARAMETER.Value;

% Prediction
y_pred = BASIS*w_infer;
plot(X, y_pred, 'r');
legend({'data', 'prediction'});
title('1d RVM Regression');







