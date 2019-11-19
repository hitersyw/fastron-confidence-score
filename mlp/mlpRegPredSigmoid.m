function Y = mlpRegPredSigmoid(model, X)
% Multilayer perceptron regression prediction
% tanh activation function is used.
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
b = model.b;
T = length(W);
Y = X;
for t = 1:T-1
    Y = sigmoid(W{t}'*Y+b{t});
end
Y = sigmoid(W{T}'*Y+b{T});