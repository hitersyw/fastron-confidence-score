close all; clear;
rng(0);

dataset = 'reachability_score';
n = [1125, 2925, 4693,  8125];
t = 10; % # number of times to run this over;
xtick = [2000, 4000, 6000, 8000, 10000];
ogle
n_models = 2;
t_train = zeros(n_models, size(n,2));
t_test = zeros(n_models, size(n,2));
mse = zeros(n_models, size(n,2));
Sn = zeros(n_models, size(n,2));

for j = 1:t
    for i = 1:numel(n)
         k = n(i);
         % SVR;
         [svr_t1, svr_t2, svr_l, svr_Sn] = svr_benchmark(dataset, k);
         t_train(1, i) = svr_t1 + t_train(1, i);
         t_test(1, i) = svr_t2 + t_test(1, i);
         mse(1, i) = svr_l + mse(1, i);
         Sn(1, i) = svr_Sn + Sn(1, i);

         % RVM;
         [rvm_t1, rvm_t2, rvm_l, rvm_Sn] = rvm_benchmark(dataset, k);
         t_train(2, i) = rvm_t1 + t_train(2, i);
         t_test(2, i) = rvm_t2 + t_test(2, i);
         mse(2, i) = rvm_l + mse(2, i);
         Sn(2, i) = rvm_Sn + Sn(2, i);

         % MLP
    %      [mlp_t1, mlp_t2, mlp_l] = mlp_benchmark(dataset, k);
    %      t_train(3, i) = mlp_t1;
    %      t_test(3, i) = mlp_t2;
    %      mse(3, i) = mlp_l;
    end
end

t_train = t_train / t;
t_test = t_test / t;
mse = mse/t;
Sn = Sn/t;


%% Training time;
figure('NumberTitle', 'off', 'Name', strrep(dataset, '_', ' ')); clf
subplot(2,2,1);
l11 = plot(n, t_train(1, :), 'b-', 'LineWidth', 1);
hold on;
l21 = plot(n, t_train(2, :), 'r-', 'LineWidth', 1);
% hold on;
% l31 = plot(n, t_train(3, :), 'g-', 'LineWidth', 1);
xlabel('Number of data points');
ylabel('Seconds');
xticks(xtick);
legend([l11, l21], {'SVR', 'RVM'},  'location','northwest');
title("Training time");

%% Test time;
subplot(2,2,2);
l12 = plot(n, t_test(1, :).*10^6, 'b-', 'LineWidth', 1);
hold on;
l22 = plot(n, t_test(2, :).*10^6, 'r-', 'LineWidth', 1);
% hold on;
% l32 = plot(n, t_test(3, :).*10^6, 'g-', 'LineWidth', 1);
xlabel('Number of data points');
ylabel('Microseconds');
xticks(xtick);
legend([l12, l22], {'SVR', 'RVM'},  'location','northwest');
title("Test time");

%% MSE;
subplot(2,2,3);
l13 = plot(n, mse(1, :), 'b-', 'LineWidth', 1);
hold on;
l23 = plot(n, mse(2, :), 'r-', 'LineWidth', 1);
% hold on;
% l33 = plot(n, mse(3, :), 'g-', 'LineWidth', 1);
xlabel('Number of data points');
xticks(xtick);
legend([l13, l23], {'SVR', 'RVM', 'MLP'},  'location','northwest');
title("MSE");

%% Support points;
subplot(2,2,4);
l14 = plot(n, Sn(1, :), 'b-', 'LineWidth', 1);
hold on;
l24 = plot(n, Sn(2, :), 'r-', 'LineWidth', 1);
xlabel('Number of data points');
xticks(xtick);
legend([l14, l24], {'SVR', 'RVM'},  'location','northwest');
title("Support points");
