% Predict labels for test inputs;
function [y_pred] = rvmPredict(K, PARAMETER)
    % Get model params;
    w_infer	= zeros(size(K,2),1);
    nz_ind = PARAMETER.Relevant;
    w_infer(nz_ind)	= PARAMETER.Value;

    % Predict;
    y_pred = K(:, nz_ind)*w_infer(nz_ind);
end