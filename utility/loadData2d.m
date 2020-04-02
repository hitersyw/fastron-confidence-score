% Load the data point from a 2D dataset
function [X, y] = loadData2d(filename)
    data = dlmread(filename,',',0,0);
    X = data(:, 1:2);
    y = data(:, 3);
end