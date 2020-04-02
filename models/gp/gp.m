function [mu, cov] = gp(Kt, Kn, Kv, y, sigma)
% Implementaion of Gaussian Process; Kt is the kernel of both training 
% points; Kn = K(x, x'), where x is in the training set, and x' is in
% the test set. Kv is the kernel of the test set. y is the values for 
% training set, and sigma is the noise parameter. 
Ky = Kt + sigma.*eye(size(y, 1));
mu = Kn' * (Ky \ y);
cov = Kv - Kn' * (Ky \ Kn);
end 