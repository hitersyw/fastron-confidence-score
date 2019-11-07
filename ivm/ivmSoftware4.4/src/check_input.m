function [params] = check_input(params)

% description: checks input of Import Vector Machine classifier
% author: Ribana Roscher (rroscher@uni-bonn.de)
% date: June 2012 (last modified)

%% check parameters
if ~isfield(params, 'Nadd')
    params.Nadd = inf;
end

if ~isfield(params, 'output')
    params.output = 1;
end

if ~isfield(params, 'maxIter')
    params.maxIter = inf;
end

if ~isfield(params, 'epsilon')
    params.epsilon = 0.001;
end

if ~isfield(params, 'delta_k')
    params.delta_k = 3;
end

if ~isfield(params, 'tempInt')
    params.tempInt = 0.95;
end

if ~isfield(params, 'epsilonBack')
    params.epsilonBack = 0.001;
end

if ~isfield(params, 'flyComputeK')
    params.flyComputeK = 0;
end

% set initial regularization parameter
if ~isfield(params, 'lambda') && ~isfield(params, 'lambda_start')
    error('choose a regularization parameter lambda or a range [lambda_start lambda_end] for searching an optimal regularization parameter')
end

if ~isfield(params, 'lambda') && ~isfield(params, 'lambda_end')
    error('choose a regularization parameter lambda or a range [lambda_start lambda_end] for searching an optimal regularization parameter')
end

if ~isfield(params, 'sigma') && ~isfield(params, 'sigmas')
    error('choose a kernel parameter sigma or a several sigmas for searching an optimal parameter')
end