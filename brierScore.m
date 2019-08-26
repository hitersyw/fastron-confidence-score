function [s] = brierScore(p,y)
    y(y==-1)=0;
    s = ((p-y).' * (p-y))/size(p,1);
end
