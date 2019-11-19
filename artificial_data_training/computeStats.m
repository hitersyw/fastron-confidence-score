n = 10;
stats = zeros(6,5);
for i = 1:n
    [acc, tpr, fpr, conf, scores] = trainModels(...
        sprintf("./data/testset/CSpace%d", i), 1, 0, 0, 0, 0);
    stats = stats + [acc; tpr; fpr; conf; scores{1}; scores{2}];
end

stats = stats / n;
% Accuracy Plots;
fig = figure();
subplot(3,1,1);
h1 = bar(stats(1:3, :)');
set(gca,'xticklabel', ["Fastron" "LogReg" "MLP" "IVM" "Bagging"]); 
l = cell(1,3);
l{1}='Accuracy'; l{2}='TPR'; l{3}='FPR';  
legend(h1,l);
title("Accuracy");

% Confidence Region plot;
subplot(3,1,2);
h2 = bar(stats(4, :)');
set(gca,'xticklabel', ["Fastron" "LogReg" "MLP" "IVM" "Bagging"]); 
title("Confidence region");

% Confidence Score plots;
subplot(3,1,3);
h3 = bar(stats(5:6, :)');
set(gca,'xticklabel', ["Fastron" "LogReg" "MLP" "IVM" "Bagging"]); 
l = cell(1,2);
l{1}='NLL'; l{2}='Brier Score';
legend(h3,l);
title("Confidence Scores");

formatOut = 'yy-mm-dd-HH-MM-SS';
saveas(gca, sprintf("./Results/%s.png", datestr(now,formatOut)));



