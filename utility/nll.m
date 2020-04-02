% Compute the negative log likelihood loss of predicited probability p,
% given label y
function [s] = nll(p, y)
    y(y==-1)=0;
    p = clip(p, 0.001);
    l = y.*log(p)+(1-y).*log(1-p);
    s = -sum(l)/size(y,1);
end