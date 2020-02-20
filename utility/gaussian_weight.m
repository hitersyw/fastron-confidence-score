function y = gaussian_weight(m, s, X)
    p1 = 1.0 / (sqrt(2*pi) * s);
    p2 = exp(-(X - m).^2 / (2 * s.^2));
    y = p1.* p2; 
end