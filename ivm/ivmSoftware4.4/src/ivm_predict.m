% IVM_PREDICT classify data with the trained Import Vector Machine model
%
%   IVM_PREDICT classify PHIT and evaluate regarding CT based on the model
%   MODEL; the result is stored in RESULT
%
%   Authors:
%     Ribana Roscher(ribana.roscher@fu-berlin.de)
%   Date: 
%     November 2014 (last modified)

function result = ivm_predict(model, phit, ct)

%% check dimensions
if size(ct, 1) ~= size(phit, 2)
    ct = ct';
end
if numel(ct) ~= size(phit, 2)
    error('dimension of test feature matrix and test label vector does not fit')
end

%% kernel
maxV = 1 - 10^-8;
minV = 10^-8;

expAlphaKs = zeros(model.C, size(phit, 2));

%% probabilities (compute chunk-wise if matrix is too large)
timestart = cputime;
N_p = 100000;
if size(phit, 2) > N_p
    S = floor(size(phit, 2) / N_p);
    result.P = [];
    for s = 1 : S
        K_test = compute_kernel(phit(:, (s-1) * N_p + 1: s * N_p), model.IV, model.kernelSigma);
        P = min(exp(model.alpha' * K_test'), eps(realmax)) ./ (ones(model.C) * min(exp(model.alpha' * K_test'), eps(realmax)));
        result.P = [result.P, P];
    end
    K_test = compute_kernel(phit(:, s * N_p+ 1:end), model.IV, model.kernelSigma);
    expAlphaKs = zeros(model.C, size(K_test, 1));
    for k = 1:model.C
        expAlphaKs(k, :) = exp(model.alpha(:, k)' * K_test')';
    end
    P = max(min(expAlphaKs ./ repmat(sum(expAlphaKs), model.C, 1), maxV), minV);
    result.P = [result.P, P];
else
    K_test = compute_kernel(phit, model.IV, model.kernelSigma);
    for k = 1:model.C
        expAlphaKs(k, :) = exp(model.alpha(:, k)' * K_test')';
    end
    result.P = max(min(expAlphaKs ./ repmat(sum(expAlphaKs), model.C, 1), maxV), minV);
end
timeend = cputime;

if any(isnan(result.P(:)))
    disp('Warning: There are nan in the probabilities.')
end

%% store results
result.trainTime = model.trainTime;
result.testTime = timeend - timestart;
result.model = model;
result.nIV = size(model.IV, 2);
result.Ntest = 0;
result.oa = 0;
result.aa = 0;
result.aa_c = 0;
result.kappa = 0;

%% accuracy (if test labels are given)
[val, idx] = max(result.P);

% if not all labels are given
if any(ct ~= 0) && ~all(ct ~= 0)
    disp('Warning: only nonzero labels are tested.')
end
    
if any(ct ~= 0)
    % find nonzero entries
    ind = find(ct);
    idx = idx(ind);
    ct = ct(ind);
    
    % confusion matrix (rows = true labels, columns = estimated labels)
    confMat = zeros(model.C, model.C);
    for i = 1 : numel(ct)
        confMat(ct(i), idx(i)) = confMat(ct(i), idx(i)) + 1;
    end

    % overall accuracy
    oa = trace(confMat) / sum(confMat(:));
    
    % class accuracy
    aa_c = diag(confMat) ./ sum(confMat,2);
    
    % average class accuracy
    aa = sum(aa_c) / length(aa_c);

    % kappa coefficient (http://kappa.chez-alice.fr/kappa_intro.htm)
    Po = oa;
    Pe = (sum(confMat) * sum(confMat,2)) / (sum(confMat(:))^2);
    K = (Po - Pe) / (1 - Pe);
    
    %% store results
    result.confMat = confMat;
    result.confMatNorm = confMat ./ repmat(sum(confMat, 2), 1, model.C);
    result.oa = oa;
    result.aa = aa;
    result.aa_c = aa_c;
    result.kappa = K;
    result.Ntest = sum(confMat(:));
    
else
    disp('Warning: no valid test samples.')
end

