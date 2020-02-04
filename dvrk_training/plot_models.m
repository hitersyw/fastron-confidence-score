close all; clear;
init;

dataset = 'reachability_score';
n = [1053, 2925, 4725, 8959];
t = 10; % # number of times to run this over;
xtick = [2000, 4000, 6000, 8000, 10000];
n_models = 3;
t_train = zeros(n_models, t, size(n,2));
t_test = zeros(n_models, t, size(n,2));
mse = zeros(n_models, t, size(n,2));
Sn = zeros(n_models, t, size(n,2));

for i = 1:t
    rng(i); % Plot with different seed;
    
    for j = 1:numel(n)
         k = n(j);
         % SVR;
         [svr_t1, svr_t2, svr_l_train, svr_l_test, svr_Sn] = svr_benchmark(base_dir, dataset, k);
         t_train(1, i, j) = svr_t1;
         t_test(1, i, j) = svr_t2;
         mse(1, i, j) = svr_l_train;
         Sn(1, i, j) = svr_Sn;

         % RVM;
         [rvm_t1, rvm_t2, rvm_l_train, rvm_l_test, rvm_Sn] = rvm_benchmark(base_dir, dataset, k);
         t_train(2, i, j) = rvm_t1;
         t_test(2, i, j) = rvm_t2;
         mse(2, i, j) = rvm_l_train;
         Sn(2, i, j) = rvm_Sn;

         % MLP
         [mlp_t1, mlp_t2, mlp_l_train, mlp_l_test] = mlp_benchmark(base_dir, dataset, k);
         t_train(3, i, j) = mlp_t1;
         t_test(3, i, j) = mlp_t2;
         mse(3, i, j) = mlp_l_train;
         
         % MLP has no support points
    end
end

% t_train = t_train / t;
% t_test = t_test / t;
% mse = mse/t;
% Sn = Sn/t;


%% Training time;
figure('NumberTitle', 'off', 'Name', strrep(dataset, '_', ' ')); clf
subplot(2,2,1);
% l11 = plot(n, t_train(1, :), 'b-', 'LineWidth', 1);
shadedErrorBar(n,squeeze(t_train(1, :, :)),{@mean,@std},'lineprops','-b','patchSaturation',0.33)
hold on;
shadedErrorBar(n,squeeze(t_train(2, :, :)),{@mean,@std},'lineprops','-r','patchSaturation',0.33)
hold on;
shadedErrorBar(n,squeeze(t_train(3, :, :)),{@mean,@std},'lineprops','-g','patchSaturation',0.33)
xlabel('Number of data points');
ylabel('Seconds');
xticks(xtick);
legend(['SVR'; 'RVM'; 'MLP'],  'location','northwest');
legend;
title("Training time");

%% Test time;
subplot(2,2,2);
shadedErrorBar(n,squeeze(t_test(1, :, :)),{@mean,@std},'lineprops','-b','patchSaturation',0.33);
hold on;
shadedErrorBar(n,squeeze(t_test(2, :, :)),{@mean,@std},'lineprops','-r','patchSaturation',0.33);
hold on;
shadedErrorBar(n,squeeze(t_test(3, :, :)),{@mean,@std},'lineprops','-g','patchSaturation',0.33);
xlabel('Number of data points');
ylabel('Seconds');
xticks(xtick);
legend(['SVR'; 'RVM'; 'MLP'],  'location','northwest');
legend;
title("Test time");

%% MSE;
subplot(2,2,3);
shadedErrorBar(n,squeeze(mse(1, :, :)),{@mean,@std},'lineprops','-b','patchSaturation',0.33);
hold on;
shadedErrorBar(n,squeeze(mse(2, :, :)),{@mean,@std},'lineprops','-r','patchSaturation',0.33);
hold on;
shadedErrorBar(n,squeeze(mse(3, :, :)),{@mean,@std},'lineprops','-g','patchSaturation',0.33);
xlabel('Number of data points');
xticks(xtick);
legend(['SVR'; 'RVM'; 'MLP'],  'location','northwest');
title("MSE");

%% Support points;
subplot(2,2,4);
shadedErrorBar(n,squeeze(Sn(1, :, :)),{@mean,@std},'lineprops','-b','patchSaturation',0.33);
hold on;
shadedErrorBar(n,squeeze(Sn(2, :, :)),{@mean,@std},'lineprops','-r','patchSaturation',0.33);
xlabel('Number of data points');
xticks(xtick);
legend(['SVR'; 'RVM'],  'location','northwest');
title("Support points");
