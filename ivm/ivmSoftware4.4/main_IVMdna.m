%MAIN_IVMDNA IVM demo file
%
%   MAIN_IVMDNA loads the data dna and runs the IVM;
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

%% dna data
load example_data/dna

%% IVM
% init parameters
params      = init;
params.Nadd = 50;

% with kernel and regularization parameter
params.output   = 1;
params.sigma    = 12;
params.lambda   = exp(-11);

% learn model
data.phi = phi;
data.phit = phi_t;
data.c = c;
data.ct = c_t;

% learn and predict model
result = ivm(data, params);
