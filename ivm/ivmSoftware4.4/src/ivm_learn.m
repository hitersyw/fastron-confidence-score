% IVM_LEARN trains the Import Vector Machine classifier
%
%   IVM_LEARN trains a IVM classification model MODEL given the following
%   input: (MxN)-dimensional features PHI with M being the feature dimension
%   and N being the number of samples; (Nx1)-dimensional label vector C
%   filled with labels 1,...,C with C being the number of classes; PARAMS
%   is a struct initialized in init.m containing the relevant
%   hyperparameters
%   
%   Authors:
%     Ribana Roscher(ribana.roscher@fu-berlin.de)
%   Date: 
%     November 2014 (last modified)

function model = ivm_learn(phi, c, params)

% description: trains the Import Vector Machine classifier
params.tempInt = 1;
params.epsilonBack = 10^-6;

%% check input
[params] = check_input(params);

maxV = 1 - 10^-8;
minV = 10^-8;

warning off

%% split sets for crossvalidation
if ~isfield(params, 'lambda')
    % choose subset for the determination of the regularization parameter
    data.phi = phi;
    data.c = c;
    [indTest indTrain] = get_CVsets(data, 5);
    phi      = data.phi(:, indTrain{1});
    c        = data.c(indTrain{1});
    model.indTrain = indTrain{1};
    
    lambda = params.lambda_start;
else
    model.indTrain = 1:size(phi, 2);
    lambda = params.lambda;
end

sigma = params.sigma;

%% dimensions and target vectors
% sizes
N = size(phi, 2);
if isfield(params, 'C')
    C = params.C;
else
    C = max(c);
end
params.Nadd = min([N, params.Nadd]);

% targets
t = labels2targets(c, C);

%% initialization
S = zeros(1, 0);        % empty subset S
KS = zeros(N, 0);       % kernel matrix of the subset S
Kreg = zeros(0,0);      % regularization matrix of the subset S
alpha = zeros(0, C);    % parameters
y = 1 / C * ones(C, N); % probabilities
W = cell([C, 1]);
z = zeros(N, C);
invHess = cell([1, C]);
all_delIV = [];

if ~params.flyComputeK
    K = compute_kernel(phi, phi, sigma);
end

%% pre-computations
% weighting matrix
for k = 1:C
    W{k}  = spdiags((y(k, :) .* (1 - y(k, :)))', 0, N, N);
end

% training error
[val_train idx_train] = max(y); %#ok<*ASGLU>
err_train = sum(c ~= idx_train');

% negative log likelihood
if numel(S) > 1
    curr_reg = lambda / 2 * sum(sum(alpha' * Kreg .* alpha'));
    lastL = -1 / N * sum(sum(t .* log(y))) + curr_reg;
else
    curr_reg = 0;
    lastL = inf;
end

y_store = y;
W_store = W; 
fval = lastL;

%% iterated reweighted least squares optimization with greedy selection
timestart1 = cputime;
for Q = 1 : params.maxIter
    % compute z from actual alpha
    tic();
    for k = 1:C
        z(:, k) = (KS * alpha(:, k) + W{k}^-1 * (t(k, :) - y(k, :))') / N;
    end

    % choose points to test to be in the subset
    candidates = setdiff(1 : N, [S, all_delIV]);
    
    if isinf(params.Nadd) || numel(candidates) == 1
       points = candidates;     
    else
       points = randsample(candidates, min([params.Nadd, numel(candidates)]));   
    end
    
    % greedy selection
    bestN = 0;
    if numel(points) > 0
        alpha_old = alpha;
        if Q > C
             % select an IV if the objective function value can be
             % decreased
            if ~params.flyComputeK
                [bestN bestval] = test_importPoints(K, KS, Kreg, y', z, points, S, lambda, ...
                        t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, lastL);
            else
                bestN = test_importPoints_fly(phi, KS, Kreg, y', z, points, S, lambda, ...
                        sigma, t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, lastL);
            end
        else
            % at least 2 import vectors are selected
            if ~params.flyComputeK
                [bestN bestval] = test_importPoints(K, KS, Kreg, y', z, points, S, lambda, ...
                        t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, lastL);
%                 [bestN bestval] = test_importPoints_fly(phi, KS, Kreg, y', z, points, S, lambda, ...
%                         sigma, t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, inf);
            else
                bestN = test_importPoints_fly(phi, KS, Kreg, y', z, points, S, lambda, ...
                        sigma, t', cell2mat(invHess'), [alpha_old; zeros(1, C)], params.tempInt, lastL);
            end
        end

        if bestN ~= 0
            alpha_store = alpha;
            KS_store = KS;
            Kreg_store = Kreg;
            
            % extend subset
            S = [S, bestN];
            
            % kernel matrices from new subset S
            KS(:, end+1)   = compute_kernel(phi, phi(:, bestN), sigma);
            Kreg(:, end+1) = compute_kernel(phi(:, S(1:end-1)), phi(:, S(end)), sigma);
            Kreg(end+1, :) = compute_kernel(phi(:, S(end)), phi(:, S), sigma);
            
            alpha_new = zeros(numel(S), C);
            invHess = cell([1, C]);
            for k = 1:C
                % inverse Hessian         
                invHess{k} = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1;

                % new parameter
                alpha_new(:, k) = invHess{k} * KS' * W{k} * z(:, k);
            end
 
            alpha = alpha_new;
            %alpha(1:Q-1, :) = alpha_old + params.tempInt * (alpha_new(1:Q-1, :) - alpha_old);
            
            % probabilities
            y = getProbabilities(alpha, KS, maxV, minV);
            
            % weighting matrix
            for k = 1:C
                W{k}  = spdiags((y(k, :) .* (1 - y(k, :)))', 0, N, N);
            end
            
            % regularization parameter
            curr_reg = lambda / 2 * sum(sum(alpha' * Kreg .* alpha'));
        end  
    end

    % negative log-likelihood 
    fval = -1 / N * sum(sum(t .* log(y))) + curr_reg;
    fval_last = fval;

    % ratio of negative loglikelihood
    rat = abs(fval - lastL) / abs(fval);

    % remove point in case of convergence
    if (rat < params.epsilon) && (bestN ~= 0)
        
        if ~isempty(points)
            S(end) = [];
            
            % kernel matrices from subset S
            KS   = KS_store;
            Kreg = Kreg_store;
        end
        
        alpha = alpha_store;
        y = y_store;
        W  = W_store; 
        
        % training error
        [val_train idx_train] = max(y); %#ok<*ASGLU>
        err_train = sum(c ~= idx_train');

        % negative log-likelihood
        curr_reg = lambda / 2 * sum(sum(alpha' * Kreg .* alpha'));
        fval = -1 / N * sum(sum(t .* log(y))) + curr_reg;
    end

    % output
    if params.output
        if ~(rat < params.epsilon)
            fprintf('forward: %3d: %d->%.2f %%, rat:%.4f,L:%f, #IV: %s\n', ...
                Q, err_train, err_train / N, rat, fval, num2str(numel(S)));
        else
            rat = abs(fval_last - lastL) / abs(fval_last);
            fval = fval_last;
            fprintf('forward: %3d: %d->%.2f %%, rat:%.4f,L:%f, #IV: %s\n', ...
            Q, err_train, err_train / N, rat, fval_last, num2str(numel(S)));
        end
    end
    
    if ~bestN
        if params.output
            disp('No further import vector can be found.')
            break
        end
    end
    
    
    %% greedy deselection
    if params.deselect
        fval_inc = fval;
        worstN = 1;
        num_dec = 0;

        % keep last 10 deselected import vectors (for tabu-search)
        if length(all_delIV) > 20
            all_delIV(1:(length(all_delIV)-20)) = [];
        end

        while (worstN ~= 0) && (numel(S) > 6)
            worstN = 0;
            num_dec = num_dec + 1;

            set = [];
            del_set = [];

            % compute z from actual alpha
            for k = 1:C
                z(:, k) = (KS * alpha(:, k) + W{k}^-1 * (t(k, :) - y(k, :))') / N;
            end

            % delete point if function value is lower without it
            currL_del = fval_inc + params.epsilonBack * fval_inc;

            % inverse Hessian
            invHess = cell([1, C]); 
            for k = 1:C
                invHess{k} = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1;
            end

            KS_store = KS;
            Kreg_store = Kreg;
            ind = 0;

            for n = 1:numel(S)
                alpha_old = alpha;            
                ind = ind + 1;

                S_del = S;

                % find nearest point from the other class, point gets wrong
                % label
    %             try
    %                 if c(S(n)) ~= idx_train(S(n))
    %                     idx_other_c = find((c == idx_train(S(n))));
    %                     idx_other_c = setdiff(idx_other_c, [S all_delIV]);
    %                     idx_other_c(ismember(idx_other_c, del_set)) = [];
    %                     if ~isempty(idx_other_c)
    %                         [min_val min_idx] = max(KS(idx_other_c, n));
    %                         S_del(end+1) = idx_other_c(min_idx);
    %                     end
    %                 end
    %             catch
    %             end
                S_del(n) = [];

                % kernel matrices
                KS_del   = KS_store;
                Kreg_del = Kreg_store;


                % get Hessian
                if numel(S_del) ~= numel(S)
                   invHess_del = cell([1, C]);
                    for k = 1:C
                        invHess_del{k} = update_inverse(invHess{k}, n);
                    end
                    ind = 1:size(invHess_del{1}, 2);
                    ind2 = ind(~ismember(ind, n));

                    % parameters
                    alpha_del = zeros(numel(S_del), C);
                    for k = 1:C
                        alpha_del(:, k) = invHess_del{k}(ind2, ind2) * (KS_del(:, ind2)' * W{k} * z(:, k));
                    end

                    % temporal integration
                    alpha_del = alpha_old(ind2, :) + params.tempInt * (alpha_del - alpha_old(ind2, :));

                    % probabilities
                    y_del = getProbabilities(alpha_del, KS_del(:, ind2), maxV, minV);

                    % compute negative log-likelihood
                    curr_reg_del = lambda / 2 * sum(sum(alpha_del' * Kreg_del(ind2, ind2) .* alpha_del'));
                    fval_del = -1 / N * sum(sum(t .* log(y_del))) + curr_reg_del;
                else
                    % kernel of current set
                    KS_del(:, n)        = []; 
                    Kreg_del(:, n)      = []; 
                    Kreg_del(n, :)      = []; 
                    KS_del(:, end+1)    = compute_kernel(phi, phi(:, S_del(end)), sigma); 
                    Kreg_del(:, end+1)  = compute_kernel(phi(:, S_del(1:end-1)), phi(:, S_del(end)), sigma); 
                    Kreg_del(end+1, :)  = compute_kernel(phi(:, S_del(end)), phi(:, S_del), sigma); 

                    % parameters
                    alpha_del = zeros(numel(S_del), C);
                    for k = 1:C
                        alpha_del(:, k) = (1 / N * KS_del' * W{k} * KS_del + lambda * Kreg_del)^-1 * ...
                                          KS_del' * W{k} * z(:, k);
                    end

                    % temporal integration
                    alpha_old(n, :) = [];
                    alpha_old = [alpha_old; zeros(1, C)];
                    alpha_del = alpha_old + params.tempInt * (alpha_del - alpha_old);

                    % probabilities
                    y_del = getProbabilities(alpha_del, KS_del, maxV, minV);

                    % compute negative log-likelihood
                    curr_reg_del = lambda / 2 * sum(sum(alpha_del' * Kreg_del .* alpha_del'));
                    fval_del = -1 / N * sum(sum(t .* log(y_del))) + curr_reg_del;
                end

                % update currL and bestN
                if fval_del <= currL_del
                    worstN = n;
                    currL_del = fval_del;
                    set = S_del;
                end
            end
            S_old = S;
            if worstN ~= 0
                alpha_old = alpha;
                all_delIV = [all_delIV, S(worstN)];
                S = set;

                if numel(S_old) ~= numel(S)
                    KS(:, worstN)   = []; 
                    Kreg(:, worstN) = []; 
                    Kreg(worstN, :) = []; 
                    for k = 1:C
                        invHess{k} = update_inverse(invHess{k}, worstN);
                    end
                    ind = 1:size(invHess{1}, 2);
                    ind2 = ind(~ismember(ind, worstN));

                    % parameters
                    alpha = zeros(numel(S), C);
                    for k = 1:C
                        alpha(:, k) = invHess{k}(ind2, ind2) * KS' * W{k} * z(:, k);
                    end

                    % temporal integration
                    alpha = alpha_old(ind2, :) + params.tempInt * (alpha - alpha_old(ind2, :));
                else
                    % kernel of current set
                    KS(:, worstN)   = []; 
                    Kreg(:, worstN) = []; 
                    Kreg(worstN, :) = []; 
                    KS(:, end+1)    = compute_kernel(phi, phi(:, S(end)), sigma); 
                    Kreg(:, end+1)  = compute_kernel(phi(:, S(1:end-1)), phi(:, S(end)), sigma); 
                    Kreg(end+1, :)  = compute_kernel(phi(:, S(end)), phi(:, S), sigma); 

                    % parameters
                    alpha = zeros(numel(S), C);
                    for k = 1:C
                        alpha(:, k) = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1 * ...
                                          KS' * W{k} * z(:, k);
                    end

                    % temporal integration
                    alpha_old(worstN, :) = [];
                    alpha_old = [alpha_old; zeros(1, C)];
                    alpha = alpha_old + params.tempInt * (alpha - alpha_old);
                end           

                % probabilities
                y = getProbabilities(alpha, KS, maxV, minV);

                curr_reg = lambda / 2 * sum(sum(alpha' * Kreg .* alpha'));
                fval = -1 / N * sum(sum(t .* log(y))) + curr_reg;

                [val_train idx_train] = max(y); %#ok<*ASGLU>
                err_train = sum(c ~= idx_train');

                rat = abs(fval - lastL) / abs(fval);
                if params.output
                    if numel(S_old) == numel(S)
                        fprintf('backward II: %s, %d->%.2f %%, rat:%f, L:%f, #IV: %s\n', ...
                        num2str(Q), err_train, err_train / N, rat, fval, num2str(numel(S)));
                    else
                        fprintf('backward: %s, %d->%.2f %%, rat:%f, L:%f, #IV: %s\n', ...
                        num2str(Q), err_train, err_train / N, rat, fval, num2str(numel(S)));
                    end
                end
            end
        end
    
        fval_dec = fval;
        fval = fval_inc;
        rat = abs(fval - lastL) / abs(fval);
        fval = fval_dec;
    end
    
    % check for convergence
    if (rat < params.epsilon || Q == params.maxIter) && (Q > 3)
        break
    end
    
    if (fval > (lastL * 1.3)) && (Q > 5)
        break
    end

    if isnan(rat)
        break
    end
    
   % check number of remaining points
    if numel(setdiff(1:N, S)) <= params.Nadd
        params.Nadd = numel(setdiff(1:N, S));
        if params.Nadd == 0
            % break if all training points are in the subset
            break
        end
        if numel(points) == 0
            break
        end
    end

    % negative log-likelihood
    curr_reg = lambda / 2 * sum(sum(alpha' * Kreg .* alpha'));
    fval = -1 / N * sum(sum(t .* log(y))) + curr_reg;

    % probabilities
    y_store = y;
    y = getProbabilities(alpha, KS, maxV, minV);
    
    % weighting matrix
    W_store = W;
    for k = 1:C
        W{k}  = spdiags((y(k, :) .* (1 - y(k, :)))', 0, N, N);
    end

    % training error
    [val_train idx_train] = max(y); %#ok<*ASGLU>
    err_train = sum(c ~= idx_train');

    invHess = cell([1, C]);
    for k = 1:C
        % inverse Hessian         
        invHess{k} = (1 / N * KS' * W{k} * KS + lambda * Kreg)^-1;
    end

    lastL = fval;
    toc();
end
timeend1 = cputime;
    
%% return results
model.P = y;
model.params = params;
model.trainError = err_train;
model.trainTime = timeend1 - timestart1;
model.IV = phi(:, S);
model.kernelSigma = params.sigma;
model.lambda = lambda;
model.C = C;
model.c = c(S);
model.S = S;
model.nIV = numel(S);
model.alpha = alpha;
model.fval = fval;

function y = getProbabilities(alpha, KS, maxV, minV)
C = size(alpha, 2);
N = size(KS, 1);
expAlphaKs = zeros(C, N);
for k = 1:C
    expAlphaKs(k, :) = exp(alpha(:, k)' * KS')';
end
y = max(min(expAlphaKs ./ repmat(sum(expAlphaKs), C, 1), maxV), minV);