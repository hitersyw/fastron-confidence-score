function [a, S, idx] = ivmTrain3(X,y,K,lambda)
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

inv_p = 0;
Hk = inf(N,1);
for k = 1:N
    tic();
    a_temp = cell(nnz(R_mark),1);
    inv_tmp = cell(nnz(R_mark),1);
    H = zeros(N,1);
   
    % Compute it for existing support points;
    idx = find(S_mark); 
    K_a = K(:, idx);
    K_q = K(idx, idx);

    F = K_a*a(idx) + a(end);
    p = 1./(1 + exp(-F));
    p = clip(p, 0.001);
    W = p.*(1-p);
    z = (F + (1./W).*(y-p));
    KW = K_a'*diag(W);
    % calculate all H costs for each point in R
    for l = find(R_mark)'
        idx_l = [idx; l];
        K_a_l = K(:, idx_l);
        K_q_l = [K_q, K(idx, l); K(l, idx), K(l, l)]; 
        if sum(S_mark) == 0
            inv_tmp{l} = (K_a_l'*(W.*K_a_l) + lambda.*K_q_l)\eye(K_q_l);
            a_temp{l} = inv_tmp{l}.*K_a_l'*(W.*z);
        else
            b = KW*K(:,l)+lambda.*K(idx,l);
            d = K(:,l)'*(W.*K(:,l))+lambda.*K(l,l);
            s = (d-b'*inv_p*b)\eye(1);
            block_inv = [inv_p + s.*inv_p*(b*b')*inv_p, -s.*inv_p*b;
                         -s.*b'*inv_p, s];
            a_temp{l} = block_inv * K_a_l'*(W.*z);
            inv_tmp{l} = block_inv;
        end
        H(l) = -y'*K_a_l*a_temp{l} + ones(1,N)*log(1+exp(K_a_l*a_temp{l})) + lambda/2*a_temp{l}'*K_q_l*a_temp{l};
    end
    
    % B3
    l = find(R_mark);
    [~, xls] = min(H(l));
    xls = l(xls);
    
    Hk(k) = H(xls);
    inv_p = inv_tmp{xls};
    % calculate bias term
%     a(end) = mean(y([find(S_mark); xls]) - K([find(S_mark); xls],:)*a(1:N));
%     a(end) = -mean(K([find(S_mark); xls],:)*a(1:N));
    a(end) = -0.5 * (mean(K(y~=0,:)*a(1:N)) + mean(K(y==0,:)*a(1:N)));
%     a(end) = -mean(K*a(1:N));
    a([find(S_mark); xls]) = a_temp{xls};
    
    S_mark(xls) = true;
    R_mark(xls) = false;
    
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