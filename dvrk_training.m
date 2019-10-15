clear; close all;
profile on;
rng(0);

%% Hyperparameters
g = 20.;

%  Fastron hyperparams;
iterMax = 10000;
beta = 1.;
Smax = 2000;
input_type = "Binary";

%% Load dataset
dir = "/home/jamesdi1993/workspace/fastron_experimental/fastron_vrep/log";
sample_file_spec = dir + "/joint_angle_sample_X_n%d_arm%d.csv";
collision_label_spec = dir + "/collision_state_y_n%d_arm%d.csv";
arm = 1;
n = 2000;

sample_file = sprintf(sample_file_spec, n, arm);
collision_label_file = sprintf(collision_label_spec, n, arm);
X = dlmread(sample_file,',',0,0);
y = dlmread(collision_label_file,',',0,0); % 1 is incollision, -1 is collision free;
% y = (y+1)/2;
training_p = 0.8;
validation_p = 0.1;
test_p = 0.1;

% train, validation and test splits;
X_train = X(1:training_p * n, :);
y_train = y(1:training_p * n, :);
X_holdout = X(training_p*n+1:(validation_p + training_p)*n, :);
y_holdout = y(training_p*n+1:(validation_p + training_p)*n, :);
X_test = X((validation_p + training_p)*n+1:n, :);
y_test = y((validation_p + training_p)*n+1:n, :);

%% Fastron with RBF; 
tic();
[a, F, K, iter]=trainFastron(X_train, y_train, @rbf, iterMax, Smax, beta, g);
t_fastron_train = toc();
fprintf("Finsihed training for Fastron. Elasped time: %s\n", t_fastron_train);

tic();
K_test_rbf = rbf(X_test, X_train, g); % n x m; 
F_test_rbf = K_test_rbf*a;
p_rbf = 1./(1 + exp(-F_test_rbf)); % without calibration;
y_pred_fastron = sign(F_test_rbf);
t_fastron_test = toc();

%% Kernel Logistic Regression
% log_reg_file_spec = "./sgd_%s.json";
% log_reg_file=sprintf(log_reg_file_spec,dataset);
% fid = fopen(log_reg_file);
% raw = fread(fid); 
% str = char(raw'); 
% fclose(fid); 
% values = jsondecode(str);
% w_lr = values.coef;
% b_lr = values.intercept;
% 
% lr_test_output = values.test_output;
% y_pred_lr_test = lr_test_output(:,1);
% p_lr_test = lr_test_output(:, 3);
% % lr_validation_output = values.validation_output;
% % y_pred_lr_validation = data(:,1);
% % p_lr_validation = data(:, 3);
% 
% x_pos = X_test(y_pred_lr_test == 1, :);
% x_neg = X_test(y_pred_lr_test == 0, :);
% 
% subplot(2,3,3);
% scatter(x_pos(:, 1), x_pos(:,2), 8, 'r', 'filled');
% hold on;
% scatter(x_neg(:, 1), x_neg(:,2), 8, 'b', 'filled');
% title(sprintf(title_spec, "Log Regression", sum(y_pred_lr_test * 2-1==y_test)/size(y_test,1)));

%% Kernel Log Regression;
tic();
[klr,FitInfo] = fitckernel(X_train,y_train, 'Learner', 'logistic');
t_klr_train = toc();

tic()
[y_pred_klr, scores_klr] = predict(klr, X_test);
t_klr_test = toc();

p_klr = scores_klr(:, 2);

%% MLP
tic()
h = [256, 256]; % Two 256 layer MLP; 
lambda = 0.0001; %  regularization term;
y_train_mlp = (y_train + 3) / 2; % convert (-1,1) into (1,2);
[model, llh] = mlpClass(X_train',y_train_mlp',h,lambda); % default is using sigmoid function;
t_mlp_train = toc();

tic();
[y_pred_mlp, p_mlp] = mlpClassPred(model,X_test');
y_pred_mlp = (y_pred_mlp * 2 - 3)'; % convert back into (-1, 1);
p_mlp = p_mlp(2, :)'; % get the probability of inCollision;
t_mlp_test = toc();

%% BaggedTrees
tic();
B = TreeBagger(10, X_train,y_train, "InBagFraction", 1.0, ...
               'NumPredictorsToSample', 'all');
t_bagging_train = toc();
fprintf("Finsihed training for Bagging. Elasped time: %s\n", t_bagging_train);

tic();
[y_pred_bagging_test, scores_bagging_test] = predict(B, X_test);
t_bagging_test = toc();
y_pred_bagging_test = cellfun(@str2double, y_pred_bagging_test);
p_bagging  = scores_bagging_test(:, 2);

%% IVM
lambda = 5;
useUnbiasedVersion = false;

tic();
K = rbf(X_train, X_train, g);
y_train_ivm = (y_train + 1) / 2; % IVM expects input to be 0, 1
if useUnbiasedVersion
    [a_ivm, S, idx] = ivmTrain(X_train, y_train_ivm, K, lambda);
else
    [a_ivm, S, idx] = ivmTrain2(X_train, y_train_ivm, K, lambda);
end
t_ivm_train = toc();
fprintf("Finsihed training for IVM. Elasped time: %s\n", t_ivm_train);

tic();
if useUnbiasedVersion
    F_test_IVM = rbf(X_test, S, g)*a_ivm;
else
    F_test_IVM = rbf(X_test, S, g)*a_ivm(1:end-1) + a_ivm(end);
end
p_ivm = 1./(1 + exp(-F_test_IVM));
t_ivm_test = toc();

y_pred_ivm = sign(F_test_IVM);

%TODO: Figure out how to do scaling with active learning strategy;

%% Bar graph for scores;
scoreFuns = {@nll, @brierScore};
names = ["Negative Log Loss", "Brier Score"];
score = zeros(5,2);
spec = "%2f";

figure_title = "%s input for %d samples";
figure('NumberTitle', 'off', 'Name', sprintf(figure_title, input_type, size(X_train,1)));
p = [p_rbf(:) p_ivm(:) p_bagging(:) p_mlp(:) p_klr(:)];
for i=1:size(p,2)
    for j=1:size(scoreFuns,2)
        scoreFun = scoreFuns{j};
        score(i,j) = scoreFun(p(:, i),y_test);
    end
end

fprintf(spec, score);

subplot(3,1,1);
h1 = bar(score);
set(gca,'xticklabel', ["Fastron" "IVM" "Bagging" "MLP" "KLR"]); 
l1 = cell(1,2);
l1{1}='Negative Log Loss'; l1{2}='Brier Score';  
legend(h1,l1);
title("Confidence Loss");

%% Matrix for calculating accuracy
results = zeros(size(X_test,1),5,4); 
counts = [y_pred_fastron y_pred_ivm y_pred_bagging_test y_pred_mlp y_pred_klr]; 

for i=1:size(results,2)
    results(:,i,1) = (counts(:,i) == 1) & (y_test == 1);
    results(:,i,2) = (counts(:,i) == 1) & (y_test ~= 1);
    results(:,i,3) = (counts(:,i) ~= 1) & (y_test ~= 1);
    results(:,i,4) = (counts(:,i) ~= 1) & (y_test == 1);
end
acc = sum(results(:,:,1) + results(:,:,3), 1)/size(y_test, 1);
tpr = sum(results(:,:,1),1)./(sum(results(:,:,1),1) + sum(results(:,:,2),1));
tnr = sum(results(:,:,3),1)./(sum(results(:,:,3),1) + sum(results(:,:,4),1));
recall_p = sum(results(:,:,1),1)./(sum(results(:,:,1),1) + sum(results(:,:,4),1));
recall_n = sum(results(:,:,3),1)./(sum(results(:,:,3),1) + sum(results(:,:,2),1));

conf = sum(p > 0.8 | p < .2, 1)/size(y_test,1);
% conf_uncalibrated = conf(:,1:2:9);
% conf_calibrated = conf(:,2:2:10);

%% Bar graph for confidence region
% figure('NumberTitle', 'off');
subplot(3,1,2);
h2 = bar(conf);
set(gca,'xticklabel', ["Fastron" "IVM" "Bagging" "MLP" "KLR"]); 
ylabel('Percentage');
title("High Confidence Region >.8 & <.2");

%% Bar graph for accuracy, precision, recall
% figure('NumberTitle', 'off');
subplot(3,1,3);
h3 = bar([acc', tpr', recall_p']);
set(gca,'xticklabel', ["Fastron" "IVM" "Bagging" "MLP" "KLR"]);
l3 = cell(1,3);
l3{1}='Accuracy'; l3{2}='Precision'; l3{3}='Recall';
ylabel('Percentage');
legend(h3, l3);
title("Accuracy, precision, recall");

%% Bar graph for training and testing time;
figure('NumberTitle', 'off', 'Name', sprintf(figure_title, input_type, size(X_train,1)));
t1 = [t_fastron_train, t_ivm_train, t_bagging_train, t_mlp_train t_klr_train];
% figure('NumberTitle', 'off');
subplot(2,1,1);
h4 = bar(t1);
set(gca,'xticklabel', ["Fastron" "IVM" "Bagging" "MLP" "KLR"]); 
ylabel("Seconds");
title("Training time");

t2 = [t_fastron_test, t_ivm_test, t_bagging_test, t_mlp_test, t_klr_test] * 1000;
subplot(2,1,2);
h5 = bar(t2);
set(gca,'xticklabel', ["Fastron" "IVM" "Bagging" "MLP" "KLR"]); 
ylabel("Microseconds");
title("Execution time");

%% Table for accuracy, precision, recall;
import mlreportgen.report.*
import mlreportgen.dom.*

T = table(round(acc',2),round(tpr',2),round(recall_p',2),round(tnr', 2),...
    round(recall_n',2), 'RowNames', ...
    ["Fastron","IVM","Bagged Trees","MLP", "KLR"],...
    'VariableNames',{'accuracy', 'tpr','recall_pos','tnr','recall_neg'
    });

t1 = BaseTable(T);
t1.Title = 'Timing results';


% uitable('Data',T{:,:},'ColumnName', T.Properties.VariableNames,...
%     'RowName',T.Properties.RowNames);
% Get the table in string form.
% TString = evalc('disp(T)');
% % Use TeX Markup for bold formatting and underscores.
% TString = strrep(TString,'<strong>','\bf');
% TString = strrep(TString,'</strong>','\rm');
% TString = strrep(TString,'_','\_');
% 
% % Get a fixed-width font.
% FixedWidth = get(0,'FixedWidthFontName');

%% Generate report;
rpt = Report('Confidence model comparisons');
chapter = Chapter();
chapter.Title = 'Timing and accuracy results';
add(rpt,chapter);
add(rpt,t1);

rptview(rpt);
% fig = figure();
% % Output the table using the annotation command.
% annotation(fig,'Textbox','String',TString,'Interpreter','Tex',...
%     'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);

