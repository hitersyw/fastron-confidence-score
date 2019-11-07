function params = init()

% description: initialization file for Import Vector Machines
% author: Ribana Roscher (rroscher@uni-bonn.de)
% date: June 2012 (last modified)

% if you want crossvalidation comment params.sigma and/or params.lambda

%% parameters for IVM
% kernel parameter
% params.sigma = 0.2;

% regularization parameter
% params.lambda = exp(-14);

% maximum number of points tested for adding to the subset (inf takes all
% points)
% default: inf
params.Nadd = 150;

% display output (0: no output, 1: output)
% default: 0;
params.output = 0;

% maximum number of iterations (maximum number of import vectors, inf
% tests all points)
% default: inf
params.maxIter = inf;

% stopping criterion for convergence proof
% default: 0.001
params.epsilon = 0.001;

% interval for computing the ratio of the negative loglikelihood
% default: 1
params.delta_k = 1;

% temporal filtering, 1: no filtering, 0.5: average between old and new
% parameters (params.tempInit < 1 is more stable, but converges slower)
% default: 0.95
params.tempInt = 0.95;

% threshold on the function value for backward selection, if an import 
% vector is removed
% default: 0.001
params.epsilonBack = 0.001;

% compute kernel on the fly (use, if kernel is too large to compute it at
% once
% default: 0
params.flyComputeK = 0;

% skip backward selection of import vector (computional cost descrease
% significantly)
% default = 0;
params.deselect = 0;

% number of folds in crossvalidation
% default = 5;
params.CV = 5;

%% params for crossvalidation
% sigmas to be tested
% default: sqrt(1 ./ (2 * 2.^(-10:1:3)))
params.sigmas = sqrt(1 ./ (2 * 2.^(-5:1:3))); 

% regularization parameter to be tested in grid search
% default: params.lambdas = [exp(-12), exp(-10:-3)]
params.lambdas = [exp(-12), exp(-10:-3)];
