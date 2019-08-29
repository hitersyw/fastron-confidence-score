function [out] = clip(p, eps)
    out = max(min(p,1-eps),eps);
end