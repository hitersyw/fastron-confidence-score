clear; close all;

% load reachability_score_n1125.mat; % bad fit
load reachability_score_n2925.mat;
% load reachability_score_n4693.mat;
% load reachability_score_n8125.mat;
% load collision_score_n1125.mat; % bad fit
% load collision_score_n2925.mat;

if exist('reachability_score', 'var')
    score_type = 'reachability';
    score = reachability_score;
elseif exist('collision_score', 'var')
    score_type = 'collision';
    score = collision_score;
end

X = score;
y = X(:, end); % targets
X = X(:, [1 2 4]); % x y theta
if 1
    % normalize between -1 and 1. works better for some reason
    scale = @(x) 2*(x - min(score(:,[1 2 4])))./(max(score(:,[1 2 4])) - min(score(:,[1 2 4]))) - 1;
else
    % normalize between 0 and 1
    scale = @(x) (x - min(X))./(max(X) - min(X));
end
X = scale(X);

svm = fitrsvm(X,y,'KernelFunction','rbf','KernelScale',0.1,'BoxConstraint',5);
y_pred = svm.predict(X) - 0.1; % not sure why there is an offset of 0.1

%%
scale_output = @(y) 1/max(y_pred)*max(y,0); 

y_pred_scaled = scale_output(y_pred);

% plot scores
if 0
    figure('Position', [44 461 570 450]);
    plot(y); hold on;
    plot(y_pred);
    plot(y_pred_scaled);
    legend('Targets', 'Predictions', 'Scaled Predictions');
end

%% plot training set and a test set
figure('Position', [327 87 1002 450]);
cm = jet(256);
thres = 0.1;

subplot(1,2,1)
colorScatter3(X(y>thres,1),X(y>thres,2),X(y>thres,3),y(y>thres), cm);
view([153 58]); axis square; grid on;
title(sprintf(''));
% make test set and plot
X_t = (max(score(:,[1 2 4])) - min(score(:,[1 2 4]))).*rand(10000,size(X,2)) ...
    + min(score(:,[1 2 4]));
X_t = scale(X_t);
y_t = scale_output(svm.predict(X_t) - 0.1);

subplot(1,2,2)
colorScatter3(X_t(y_t>thres,1),X_t(y_t>thres,2),X_t(y_t>thres,3),y_t(y_t>thres), cm);
view([153 58]); axis square; grid on;
%%
figure('Position', [1000 530 560 420]);

edges = linspace(0,1,10);
h = histcounts(y,edges);
h_pred = histcounts(y_pred,edges);
h_pred_scaled = histcounts(y_pred_scaled,edges);
h_t =  histcounts(y_t,edges);
bar(edges(1:end-1),[h; h_pred; h_pred_scaled; h_t]'./[length(y)*ones(1,3) length(y_t)]);

legend('Targets', 'Predictions', 'Scaled Predictions', 'Scaled Test Predictions');