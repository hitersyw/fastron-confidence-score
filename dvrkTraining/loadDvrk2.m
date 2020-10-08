function [X, y] = loadDvrk2(input_path, dataset, n, show_dist)
    % Load a Dvrk dataset. 
    % 
    % Input:
    %    input_path  - the directory and file specification for the dataset
    %    dataset     - one of ['reachability', 'self_collision',
    %    'env_collision']
    %    n           - number of samples from the dataset
    %    show_dist   - plot the distribution of the dataset 
    score_dict = load(sprintf(input_path, dataset, n));
    score = getfield(score_dict, dataset);
    X = score(:, [1 2 4]); % omit z because it is held constant in our dataset; [x,y,\theta]
    y = score(:, 5);
    
    if show_dist
        figure; clf;
        histogram(y, 10,'Normalization','probability');
        title(sprintf('%s, n:%d', strrep(dataset, '_', ' '), n));
    end
end