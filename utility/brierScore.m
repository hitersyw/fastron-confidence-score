% Brier score, or MSE for calculating loss of predicted probabilities. 
function [s] = brierScore(p,y)
    y(y==-1)=0;                      % Convert -1 to 0; 
    s = ((p-y).' * (p-y))/size(p,1);
end
