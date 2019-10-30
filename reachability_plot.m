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

shape = [15,15,5]; % (x, y, theta);
x1 = reshape(X(:, 1), shape);
x2 = reshape(X(:, 2), shape);
y = reshape(y, shape);
figure();

% fix theta;
plot3(reshape(x1(:, :, 1), [15*15,1]), reshape(x2(:, :, 1),[15*15,1]), reshape(y(:, :, 1),[15*15,1]), 'b.');
xlabel('X');
ylabel('Y');
zlabel('score');