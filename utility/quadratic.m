% Compute the quadratic function of a matrix X, centered at u.
function y = quadratic(a, b, u, X)
    y = -a.*((X - u).^2) + b;
end