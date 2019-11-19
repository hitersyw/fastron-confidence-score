function [out] = feedforward(X, w, b)
    in = X;
    n_layers = size(w,1);
    for i=1:n_layers-1
        out =in*w{i}+ b{i}';
        out(out<0)=0; % Relu
        in = out;
    end
    out =in*w{n_layers}+ b{n_layers}'; % last layer
end