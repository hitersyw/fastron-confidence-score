% Compute the weights for a dataset X according to a gaussian distribution,
% with mean m and standard deviation s.
function y = gaussianWeight(m, s, X)
    p1 = 1.0 / (sqrt(2*pi) * s);
    p2 = exp(-(X - m).^2 / (2 * s.^2));
    y = p1.* p2; 
end