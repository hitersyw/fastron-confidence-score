function [s] = bceLoss(p, y)
    y(y==-1)=0;
    
    l = y.*log(p)+(1-y).*log(1-p);
    l(p==y)=0;
    s = -sum(l)/size(y,1);
end