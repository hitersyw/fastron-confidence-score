clear; close all;
rng(0);
init;

%% load reachability score data;
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
score_dict = load('/home/nikhildas/workspace/fastron_experimental/fastron_vrep/constraint_analysis/log/reachability_score_n1125.mat');
score = getfield(score_dict, 'reachability_score');
X = score(:, 1:4);
y = score(:, 5);
n = size(X, 1);
input_type = "Score from 0 to 1";
spec = "%2f";

shape = [5, 15, 15]; % (theta, y, x);
x1 = reshape(X(:, 1), [5, 15, 15]);
x2 = reshape(X(:, 2), [5, 15, 15]);
theta = reshape(X(:, 4), [5, 15, 15]);
y = reshape(y, shape);

% fix theta;
for i = 1:5
    
    figure();
    surf(reshape(x1(i, :, :), [15,15]), reshape(x2(i, :, :),[15,15]), reshape(y(i, :, :),[15,15]), reshape(y(i, :, :),[15,15]));
    xlabel('X');
    ylabel('Y');
    zlabel('score');
    title(gca, sprintf('Theta: %2f', theta(i, 1, 1)));
end