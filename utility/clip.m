% Clip the values between [eps, 1 - eps]
function [out] = clip(p, eps)
    out = max(min(p,1-eps),eps);
end