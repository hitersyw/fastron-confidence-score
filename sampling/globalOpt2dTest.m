close all; clear;

n = 128; 
x = linspace(-0.5, 0.5, n);
y = linspace(-0.5, 0.5, n);
[xs, ys] = meshgrid(x, y);

f = @(x,y) 1.0 ./ (pdist2([x, y], [0, 0]) + 1.0);

V = f(xs(:), ys(:));
Vs = reshape(V, n, n);

figure();
imagesc(x, y, Vs, [0.586, 1]); hold on;
colormap('jet');

[x, y, v] = searchGlobalOpt2d(xs, ys, Vs, 4);