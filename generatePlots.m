clear; close all;

plot_fig = false;
rng(0);
n = 2;

mkdir ./data testset;
generate2Dsets(n, "./data/testset", plot_fig);
for i = 1:n
    preprocess(sprintf("./data/testset/CSpace%d", i));
end


