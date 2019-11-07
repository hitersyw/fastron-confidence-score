function t = labels2targets(labels, C)

% description: convert labels to target matrix
% author: Ribana Roscher (rroscher@uni-bonn.de)
% date: December 2010 (last modified)

%% number of data
N = size(labels(:), 1);

%% targets
t = accumarray([labels, (1 : N)'], true(N, 1), [C, N]);
