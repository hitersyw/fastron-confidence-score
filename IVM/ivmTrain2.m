function [a, S, idx] = ivmTrain2(X,y,K,lambda)
% [a, S, idx] = ivmTrain2(X,y,K,lambda) returns weight vector a, support set S,
% and indices of input data idx, given input data X, labels y, kernel
% matrix K, and regularization parameter lambda. Implementation of Zhu et
% al.'s Import Vector Machine. The last element of a is a constant bias
% term.

% B1
N = size(X, 1);
a = zeros(N+1,1);

S_mark = false(N,1);
R_mark = ~S_mark;

% B2
if nargin < 4, lambda = 1; end

Hk = inf(N,1);
for k = 1:N
    tic();
    a_temp = cell(nnz(R_mark),1);
    H = zeros(nnz(R_mark),1);
    
    i = 0; % keep track of the indices in H;
    % calculate all H costs for each point in R
    for l = find(R_mark)'
        i = i + 1;
        idx = [find(S_mark); l];
        
        K_a = K(:, idx);
        K_q = K(idx, idx);
        
        F = K_a*a(idx) + a(end);
        p = 1./(1 + exp(-F));
        p = clip(p, 0.001);
        W = p.*(1-p);
        
        z = (F + (1./W).*(y-p));
        a_temp{i} = (K_a'*(W.*K_a) + lambda.*K_q)\K_a'*(W.*z);
        
        H(i) = -y'*K_a*a_temp{i} + ones(1,N)*log(1+exp(K_a*a_temp{i})) + lambda/2*a_temp{i}'*K_q*a_temp{i};
    end
    
    % B3
    l = find(R_mark);
    [~, xls] = min(H);
    Hk(k) = H(xls);
    idx = l(xls);
    
    % calculate bias term
%     a(end) = mean(y([find(S_mark); xls]) - K([find(S_mark); xls],:)*a(1:N));
%     a(end) = -mean(K([find(S_mark); xls],:)*a(1:N));
    mean_pos = mean(K(y~=0,:)*a(1:N));
    mean_neg = mean(K(y==0,:)*a(1:N));
    mean_pos(isnan(mean_pos)) = 0;
    mean_neg(isnan(mean_neg)) = 0;
    a(end) = -0.5 * (mean_pos + mean_neg);
%     a(end) = -mean(K*a(1:N));
    a([find(S_mark); idx]) = a_temp{xls};
    
    S_mark(idx) = true;
    R_mark(idx) = false;
    
    if k > 1 && abs((Hk(k) - Hk(k-1))/Hk(k)) < 0.001 % termination condition
        break;
    end
    toc();
end
display(sprintf('Iterations: %d', k));

a = a([find(S_mark); N+1],:);
S = X(S_mark,:);
idx = find(S_mark);
end