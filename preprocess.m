% Script for preprocessing a dataset into training, validation and
% test-set;
dataset = "circle";
fractions = [.6, .2, .2]; % fraction of training, validation, test set;
names = ["training", "validation","test"];
spec = "./data/%s.txt";
output_spec = "./data/%s_%s.csv";
data_file = sprintf(spec, dataset);
data = dlmread(data_file,',',0,0);

% Permute data;
data = data(randperm(size(data, 1)), :);

% Divide data into fractions;
low = 1;
for i = 1:size(names,2)
    high = size(data, 1)*sum(fractions(1:i));
    subset=data(low:high, :);
    output_file = sprintf(output_spec, dataset, names(i));
    dlmwrite(output_file, subset);
    % writeData(output_file, total_points*total_test_p, size(x3,1), size(x4,1), test);
    low = high+1;
end