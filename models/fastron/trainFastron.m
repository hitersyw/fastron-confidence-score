function [a, F, K, iter] = trainFastron(X, y, f, iterMax, Smax, beta, g, a, F, K)
    % Train fastron on a dataset X with label y
    % 
    % Input: 
    %   X: The training data;
    %   y: The labels;
    %   iterMax: Maximum number of times to iterate through the algo;
    %   Smax: Support points limit;
    %   beta: Conditional bias for points that are negative;
    %   g: gamma, or width of the data point;
    % Output:
    %   a: The original weights for data points;
    %   F: Hypothesis function;
    %   K: Kernel matrix;
    %   iter: Number of iterations
    % Written by Nikhil Das.
    

% Train or update Fastron model

% "train Fastron from scratch" mode if not enough input arguments
% "update Fastron" mode if true.
if nargin < 8
    
    a = zeros(size(X,1),1); % weight of each point in data
    F = a; % hypothesis function we are learning, evaluated at each point of data
    K = f(X, X, g); % compute kernel
%     K = exp(-g*pdist2(X,X).^2);
else
    idx = find(a.*y < 0);
    for i = idx'
        F = F - a(i)*K(:,i);
        a(i) = 0;
    end
end

% Conditional bias parameter
beta_vec = ones(size(y)); 
beta_vec(y<0) = 1/beta;
% beta_vec = beta_vec - 1/beta;
% beta_vec(y>0) = beta;

for iter = 1:iterMax
    % Check for misclassifications
    [min_margin, i] = min(y.*F);
    if min_margin <= 0
        if (nnz(a) < Smax || a(i))
            delta_a = beta_vec(i)*y(i) - F(i);
            a(i) = a(i) + delta_a;
            F = F + delta_a*K(:,i);
            % continue to the next iteration;
            continue;
        end
        % If you reach here, you need to correct a point, but can't
    end
    
    % Remove redundant support points
    [max_margin_removed, i] = max(y.*(F-a).*(a~=0));
    if max_margin_removed > 0 && nnz(a) > 1
        F = F - a(i)*K(:,i);
        a(i) = 0;
        continue;
    end
    
    % If you reach here, don't need to add points and can't remove any
    if (nnz(a) == Smax)
        display(sprintf('FAILED: Hit support point limit at iteration %d ', iter));
        return;
    else
        display(sprintf('SUCCESS: Model update complete at iteration %d ', iter));
        return;
    end
end

% If you reach here, you hit iteration limit
display(sprintf('FAILED: Did not converge after %d iterations', iter));
return;
end