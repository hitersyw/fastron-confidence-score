function [X_psm1, y_psm1, X_psm2, y_psm2] = load_dvrk_two_arm(input_path, dataset, n, use_fastron, show_dist)
    
    if use_fastron
        fastron_suffix = "_fastron";
    else
        fastron_suffix = "";
    end
    psm1_score_dict = load(sprintf(input_path, dataset + fastron_suffix, n));
    score = getfield(score_dict, dataset);
    X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
    y = score(:, 5);
    
    if show_dist
        close all;
        figure; clf;
        histogram(y, 10,'Normalization','probability');
        title(sprintf('%s, n:%d', strrep(dataset, '_', ' '), n));
    end
end