% Rational Quadratic kernel
function [K] = rq(X1, X2, g) 
  r2 = 1+g/2*pdist2(X1, X2).^2; 
  K = 1./(r2.*r2);
end