% Script for preprocessing a dataset into training, validation and
% test-set;
function [] = preprocess(base_path)
    fractions = [.6, .2, .2]; % fraction of training, validation, test set;
    names = ["training", "validation","test"];
    output_spec = base_path + "/%s.csv";
    data_file = base_path + "/configs.txt";
    data = dlmread(data_file,',',0,0);

    % Permute data;
    data = data(randperm(size(data, 1)), :);

    % Divide data into fractions;
    low = 1;
    for i = 1:size(names,2)
        high = size(data, 1)*sum(fractions(1:i));
        subset=data(low:high, :);
        output_file = sprintf(output_spec, names(i));
        dlmwrite(output_file, subset);
        % writeData(output_file, total_points*total_test_p, size(x3,1), size(x4,1), test);
        low = high+1;
    end
end