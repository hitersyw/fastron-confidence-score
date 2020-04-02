% Fit Gaussian process on 1d sinusoidal data. 
close all; clear; 
init;

g = 20;

figure();
n = 1000;
x = linspace(-2, 2, n);
y = sin(2*pi*x);
e = rand(1, n);
z = y + e;
plot(x, z, 'b.');
hold on;


gprMdl = fitrgp(x', z', 'KernelFunction', 'squaredexponential', ...
    'Sigma', sqrt(1.0/(2*g)));
[mu, std, yint] = resubPredict(gprMdl);
plot(x, mu, 'r');
shadedErrorBar(x, mu, std);
legend({'data', 'prediction'});
title('1d Gaussian Process');
