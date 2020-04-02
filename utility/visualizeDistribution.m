% Sample script for visualizing the distribution of the predicted labels
% compared to the true distribution.

function [] = visualizeDistribution(t, y, ys)
% Inputs:
%    t - target variable
%    y - predicted variable
%    y - predicted variable scaled
edges = linspace(0, 1, 11);
h = histcounts(t,edges);
h_pred = histcounts(y,edges);
h_pred_scaled = histcounts(ys,edges);
bar(edges(1:end-1),[h; h_pred; h_pred_scaled]'./ (length(t)*ones(1,3))); %Normalization
legend('Targets', 'Predictions', 'Scaled Predictions', 'Test Predictions');
title("Target and predicted distribution");