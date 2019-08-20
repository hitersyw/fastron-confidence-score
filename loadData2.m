% Load the data points from a file.
function [X, y] = loadData2(filename)
    % Read the first three lines; 
    data = dlmread(filename,',',0,0);
    X = data(:, 1:2);
    y = data(:, 3);
end