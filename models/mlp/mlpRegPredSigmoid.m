function Y = mlpRegPredSigmoid(model, X)
% Multilayer perceptron regression prediction
% tanh activation function is used.
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Adapted from code written by Mo Chen from PRML library.
W = model.W;
b = model.b;
T = length(W);
Y = X;
for t = 1:T-1
    Y = sigmoid(W{t}'*Y+b{t});
end
Y = sigmoid(W{T}'*Y+b{T});