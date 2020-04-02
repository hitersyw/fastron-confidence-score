% Train IVM on a 2d dataset.
clear; close all;

%% create data and define labels
rng(0);

N = 400;
data = 2*rand(N,2) - 1;
y = zeros(N,1);

rule = @(X) sum(X.^2, 2) < 0.5^2; % circle
% rule = @(X) max(abs(X), [], 2) < 0.5; % square
y(rule(data)) = 1;

%% set up kernel function
Kfun = @(X,Y) exp(-5 * pdist2(X, Y).^2);
K = Kfun(data, data);

%% train IVM model
% regularization parameter
lambda = 3;

% set this to true if you want to use original (unbiased) method
% set to false for an approximated constant bias term
useUnbiasedVersion = true;

if useUnbiasedVersion
    [a, S, idx] = ivmTrainUnbiased(data, y, K, lambda);
else
    [a, S, idx] = ivmTrainBiased(data, y, K, lambda);
    
    % Uncomment this if you want to use a more efficient version of IVM; 
    % [a, S, idx] = ivmTrainEfficient(data, y, K, lambda);
end

%% calculate probabilities
[X,Y] = meshgrid(linspace(-2,2,200), linspace(-2,2,200));
p = zeros(size(X));
if useUnbiasedVersion
    F = Kfun([X(:) Y(:)], S)*a;
else
    F = Kfun([X(:) Y(:)], S)*a(1:end-1) + a(end);
end

p(:) = 1./(1 + exp(-F));

%% plot IVM model
imagesc([-2 2], [-2 2], imresize(p, 4), [0 1]); hold on;

plot(data(y~=0, 1), data(y~=0, 2), 'm.');
plot(data(y==0, 1), data(y==0, 2), 'c.');
plot(S(:,1), S(:,2), 'ko');
axis square;

%% plot contours of probabilities
[cont, cont_h] = contour(X, Y, p);
cont_h.LevelList = [0.1 0.25 0.5 0.75 0.9];
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);

title('P(Y = 1|X = x)');

cm = interp1([0 0.5 1]', [0.4039 0.6627 0.8118; 0.9686 0.9686 0.9686;0.9373 0.5412 0.3843;], linspace(0,1,256)');
colormap(cm);

%% plot true boundary in red
imgGt = zeros(size(X));
imgGt(rule([X(:) Y(:)])) = 1;
[~,cont_h] = contour(X, Y, imgGt);
cont_h.LevelList = 0.5;
cont_h.LineColor = 'r';