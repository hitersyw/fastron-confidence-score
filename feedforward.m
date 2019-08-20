function [out] = feedforward(X, w, b)
    in = X;
    out = zeros(size(X,1), size(w{1},2));
    for i=1:size(w,1)
        out = in * w{i} + b{i};
        out(out < 0)=0; % Relu
        in = out;
    end
end