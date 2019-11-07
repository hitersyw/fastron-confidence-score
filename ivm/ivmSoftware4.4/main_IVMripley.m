%MAIN_IVMRIPLEY IVM demo file
%
%   MAIN_IVMRIPLEY loads the data ripley and runs the IVM with
%   crossvalidated training and testing;
%   IVM trains the IVM model and classify test data, whereas the output is
%   stored in the struct RESULT:
%       RESULT.P: probabilities of test data
%       RESULT.trainTime: training time
%       RESULT.testTime: testing time
%       RESULT.confMat: confusion matrix (rows: true label, columns: estimated label)
%       RESULT.perc: User and Producer accuracy
%       RESULT.oa: overall accuracy
%       RESULT.aa: average accuracy
%       RESULT.aa_c: class-wise average accuracy
%       RESULT.kappa: kappa coefficient
%       RESULT.nIV: number of used import vectors
%       RESULT.Ntest: number of testing data
%       RESULT.model: stored model
%           RESULT.model.indTrain: indices of used training points
%           RESULT.model.P: probabilities of train data
%           RESULT.model.lambda: regularization parameter
%           RESULT.model.trainError: absolute training error
%           RESULT.model.params: used params (output from init)
%           RESULT.model.trainTime: training time
%           RESULT.model.IV: feature vectors of the import vectors
%           RESULT.model.kernelSigma: used kernel parameter
%           RESULT.model.lambda: used regularization parameter
%           RESULT.model.C: number of classes
%           RESULT.model.c: true labels of import vectors
%           RESULT.model.S: indices of the import vectors
%           RESULT.model.nIV: number of used import vectors
%           RESULT.model.alpha: parameters of the decision hyperplane
%           RESULT.model.fval: last function value of objective function
%           RESULT.model.CVacc: highest crossvalidation value
%   The input is the struct DATA with DATA.PHI being the (MxN)-dimensional 
%   features with M being the feature dimensionand N being the number of 
%   samples, and DATA.C being a (Nx1)-dimensional label vector C filled 
%   with labels 1,...,C with C being the number of classes; optional are 
%   DATA.PHIT being the (MxT)-dimensional testing data with M dimensions 
%   and T training samples, and DATA.CT the test labels; PARAMS is a struct
%   initialized in init.m containing the relevant hyperparameters
%   
%   Authors:
%     Ribana Roscher(ribana.roscher@fu-berlin.de)
%   Date: 
%     November 2014 (last modified)

clc
clear all
close all

addpath('src')

%% colors for plotting
load example_data/map
color = map;
color(2, :) = [0.9 0 0];
marker = {'o', 'x'};
 
%% riplay data
load example_data/ripley

% data points
X = ripley(:, 1:2)';
[D, N] = size(X);
phi = X;
M = D + 1;

% labels
c = ripley(:, 3) + 1;

%% test data for plotting
boxG	= [min(X(1, :)) max(X(1, :)) min(X(2, :)) max(X(2, :))];
gsteps          = 300;
range1          = boxG(1):(boxG(2)-boxG(1))/(gsteps-1):boxG(2);
range2          = boxG(3):(boxG(4)-boxG(3))/(gsteps-1):boxG(4);
[grid1, grid2]	= meshgrid(range1,range2);
Xgrid           = [grid1(:) grid2(:)];
phit            = Xgrid'; 

% dummy label input
ct = ones(size(phit, 2), 1)';

%% IVM
% init hyperparameters
params = init;

% set data struct
data.phi = phi;     % train data
data.phit = phit;   % test data
data.c = c;         % train labels
data.ct = ct;       % test labels

% with known kernel and regularization parameter (uncomment to run algorithm with
% known parameter and without gridsearch)
% params.sigma = 0.25;
% params.lambda = exp(-5);

% learn and predict
result = ivm(data, params);

%% plot result
figure
hold on
scatter(X(1, c == 1)', X(2, c == 1)', 20, color(1, :), 'filled')
scatter(X(1, c == 2)', X(2, c == 2)', 20, color(2, :), 'filled', 'Marker', 'x', ...
        'MarkerEdgeColor', color(2, :))
axis off

[cIV, IV] = intersect(X', result.model.IV', 'rows');

% plot import point
for iq = IV'
    scatter(X(1, iq)', X(2, iq)', 70, ...
        color(c(iq), :), 'filled', 'Marker', marker{c(iq)}, 'MarkerEdgeColor', color(c(iq), :), 'Linewidth', 2)
end

% classify test data
[P_val, P_idx] = max(result.P);

labelGridIVM = reshape(P_idx, sqrt(length(P_val)), sqrt(length(P_val)));

for i = 2 : size(labelGridIVM, 1) - 1
    for j = 2 : size(labelGridIVM, 1) - 1
        if (labelGridIVM(i,j) ~= labelGridIVM(i-1,j)) || ...
           (labelGridIVM(i,j) ~= labelGridIVM(i,j-1))        
            ind  = sub2ind(size(labelGridIVM), i, j);
            plot(grid1(ind), grid2(ind),'.k', 'Markersize', 2)
        end
    end
end

