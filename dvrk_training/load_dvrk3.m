function [X, y] = load_dvrk3(input_path, dataset, n, show_dist)
    
    score_dict = load(sprintf(input_path, dataset, n));
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