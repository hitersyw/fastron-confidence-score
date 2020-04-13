function [x, y, z, v] = searchGlobalOpt3(X, Y, Z, V, s)
% Search for the global optimum from a 3D meshdgrid [X, Y, Z], based on 
% coarse to fine sampling.
% Input:
%      X - The X dimension of meshgrid
%      Y - The Y dimension of meshgrid
%      Z - The Z dimension of meshgrid
%      V - The value function
%      s - stride of the search in the begining, must be a power of 2
% Output: 
%      x - x coordinate of the optimal point
%      y - y coordinate of the optimal point
%      z - z coordiante of the optimal point
%      v - value of the point
while s >= 1
    ns = size(X) / s; % number of points on each axis; 
    d  = floor(s / 2);       % offset from each grid;

    % Points selected
    xind = linspace(s - d, size(X,1) - d, ns(1));
    yind = linspace(s - d, size(X,2) - d, ns(2));
    zind = linspace(s - d, size(X,3) - d, ns(3));

    Vs = V(xind, yind, zind);
    [v, ind] = max(Vs, [], 'all', 'linear');
%     
%     indices = sub2ind(size(xs), xs(:)', ys(:)');
%     Vq = V(:); 
%     [v, ind] = max(Vq(indices));

    [xs, ys, zs] = meshgrid(xind, yind, zind);
    [p, q, m] = ind2sub(size(xs), ind);
    i = yind(p); j = xind(q); k = zind(m);
    x = X(i, j, k); y = Y(i, j, k); z = Z(i, j, k); v = V(i, j, k);

    fprintf("Maximum value found at (%.1f, %.1f, %.1f): %.2f, stride: %d\n", x, y, z, v, s);
%     rectangle('Position',[xmin ymin xmax-xmin ymax-ymin], 'EdgeColor','r');
%     plot(x, y, 'wo');
%     hold on;

    % Move on to the smaller window; 
    X = X(i-d+1:i+d, j-d+1:j+d, k-d+1:k+d);
    Y = Y(i-d+1:i+d, j-d+1:j+d, k-d+1:k+d);
    Z = Z(i-d+1:i+d, j-d+1:j+d, k-d+1:k+d);
    V = V(i-d+1:i+d, j-d+1:j+d, k-d+1:k+d);

    s = s / 2;
end

    