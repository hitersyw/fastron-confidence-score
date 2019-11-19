% Configuration of the grid
total_points = 2000;

x_min = [-2, -2];
x_max = [2, 2];
r = 1;

X = x_min + (x_max - x_min).*rand(total_points, 2);
dist = pdist2([0 0], X).';
y = 2.*(dist < r) - 1;
data = [X y];

filename = "./data/circle.txt";
dlmwrite(filename, data);
