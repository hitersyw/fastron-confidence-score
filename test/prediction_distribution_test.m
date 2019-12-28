close all; clear
rng(0);
init;

g = 50;
p_train = 1.0;

dataset = "reachability_score";
load reachability_score_n2925.mat;
n = 2925;
score = reachability_score;
X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
y = score(:, 5);

% scale the dataset;
if 1
    % normalize between -1 and 1. works better for some reason
    scale = @(x) 2*(x - min(x))./(max(x) - min(x)) - 1;
else
    % normalize between 0 and 1
    scale = @(x) (x - min(x))./(max(x) - min(x));
end

X = scale(X);
% shuffle the dataset;
idx = randperm(n);
X = X(idx, :);
X_train = X(1:ceil(n*p_train), :);
y_train = y(1:ceil(n*p_train));

X_test = X(ceil(n*p_train+1):n, :);
y_test = y(ceil(n*p_train+1):n);

scale_output = @(y) 1/max(y)*max(y,0); 
size = size([y_train; y_test], 1);

%% SVR;
svrMdl = svrTrain2(X_train, y_train, X_test, y_test);
y_svr = predict(svrMdl, X_train);
y_svr_scaled = scale_output(y_svr);
y_svr_test = predict(svrMdl, X_test);


close all;
figure; clf;

edges = linspace(0,1,11);
h = histcounts(y,edges);
h_pred = histcounts(y_svr,edges);
h_pred_scaled = histcounts(y_svr_scaled,edges);
h_pred_test = histcounts(y_svr_test,edges);
bar(edges(1:end-1),[h; h_pred; h_pred_scaled; h_pred_test]'./[length(y_train)*ones(1,3) length(y_test)]); %Normalization
legend('Targets', 'Predictions', 'Scaled Predictions', 'Test Predictions');