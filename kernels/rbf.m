function [K] = rbf(X1, X2, g)
  K = exp(-g*pdist2(X1,X2).^2);
end