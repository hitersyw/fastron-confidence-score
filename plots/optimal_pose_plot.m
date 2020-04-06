clear; close all;
rng(0);
init;

%% load reachability score data;
% score_dict = load("/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/"...
%  + "ConfidenceScore/constraint_log/collision_score.mat");
T = readtable('/home/jamesdi1993/workspace/arclab/fastron-confidence-score/optimal_pose_test/pose/poses_combined_validated.csv');
X = T{:, [1,2,4]};
y_reach = T{:, 6};
y_collision = T{:, 9};
combined_score = y_reach + y_collision;
thres = 0.4;
cm = jet(256);
colorScatter3(X(y_reach>thres & y_collision > thres,1),...
              X(y_reach>thres & y_collision > thres, 2),...
              X(y_reach>thres & y_collision >thres,3), ...
              combined_score(y_reach>thres & y_collision >thres),...
              cm);
view([153 58]); axis square; grid on;
title("Ground-truth Score");