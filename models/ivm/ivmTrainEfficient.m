function [a, S, idx] = ivmTrainEfficient(X, y, K, lambda)
% [a, S, idx] = ivmTrainEfficient(X,y,K,lambda) returns weight vector a, support set S,
% and indices of input data idx, given input data X, labels y, kernel
% matrix K, and regularization parameter lambda. Implementation of Zhu et
% al.'s Import Vector Machine. The last element of a is a constant bias
% term.
% 
% Adapted from ivmTrainBiased and provides a more efficient implementation

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
    H = zeros(nnz(R_mark),1);
   
    % Compute it for existing support points;
    idx = find(S_mark); 
    K_a = K(:, idx);
    K_q = K(idx, idx);

    F = K_a*a(idx) + a(end);
    p = 1./(1 + exp(-F));
    p = clip(p, 0.001);
    W = p.*(1-p);
    z = (F + (1./W).*(y-p));
    KW = K_a'*diag(W); % TODO: simplify this term;
    i = 0;
    % calculate all H costs for each point in R
    for l = find(R_mark)'
        i = i + 1;
        idx_l = [idx; l];
        K_a_l = K(:, idx_l);
        K_q_l = [K_q, K(idx, l); K(l, idx), K(l, l)]; 
        % If start is 0;
        if sum(S_mark) == 0
            inv_tmp{i} = inv(K_a_l'*(W.*K_a_l) + lambda.*K_q_l);
            a_temp{i} = inv_tmp{i}*K_a_l'*(W.*z);
        else
            b = KW*K(:,l)+lambda.*K(idx,l);
            d = K(:,l)'*(W.*K(:,l))+lambda.*K(l,l);
            s = 1./(d-b'*inv_p*b);
            block_inv = [inv_p + s.*inv_p*(b*b')*inv_p, -s.*inv_p*b;
                         -s.*b'*inv_p, s];
            a_temp{i} = block_inv * K_a_l'*(W.*z);
            inv_tmp{i} = block_inv;
        end
        H(i) = -y'*K_a_l*a_temp{i} + ones(1,N)*log(1+exp(K_a_l*a_temp{i})) + lambda/2*a_temp{i}'*K_q_l*a_temp{i};
    end
    
    % B3
    l = find(R_mark);
    [~, xls] = min(H);
    Hk(k) = H(xls);
    inv_p = inv_tmp{xls};
    
    idx = l(xls);
    % calculate bias term
%     a(end) = mean(y([find(S_mark); xls]) - K([find(S_mark); xls],:)*a(1:N));
%     a(end) = -mean(K([find(S_mark); xls],:)*a(1:N));
    mean_pos = mean(K(y~=0,:)*a(1:N));
    mean_neg = mean(K(y==0,:)*a(1:N));
    mean_pos(isnan(mean_pos))=0;
    mean_neg(isnan(mean_neg))=0;
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