function [x, y, v] = searchGlobalOpt2d(X, Y, V, s)
% Search for the global optimum from a 2D meshgrid [X, Y, Z], based on 
% coarse to fine sampling.
% Input:
%      X - The X dimension of meshgrid
%      Y - The Y dimension of meshgrid
%      V - The value function
%      s - stride of the search in the begining, must be a power of 2
% Output: 
%      x - the x coordinate of the maximum point
%      y - the y coordinate of the maximum point
%      v - the value of the max point    
    % TODO: add visualization for each point
    % TODO: add padding to make each dimension of X divisible by d;
%     assert(mod(s, 2) == 0);
%     assert(all(mod(size(X), s) == 0));        
        
    while s >= 1

        ns = size(X) / s; % number of points on each axis; 
        d  = floor(s / 2);       % offset from each grid;

        % Points selected
        xind = linspace(s - d, size(X,1) - d, ns(1));
        yind = linspace(s - d, size(X,2) - d, ns(2));
           
        Vs = V(xind, yind);
        [v, ind] = max(Vs, [], 'all', 'linear');
    %     
    %     indices = sub2ind(size(xs), xs(:)', ys(:)');
    %     Vq = V(:); 
    %     [v, ind] = max(Vq(indices));
        
        [xs, ys] = meshgrid(xind, yind);
        [row, col] = ind2sub(size(xs), ind);
        i = yind(row); j = yind(col);
        x = X(i, j); y = Y(i, j); v = V(i, j);
        
        xmin = min(X, [], 'all'); xmax = max(X, [], 'all');
        ymin = min(Y, [], 'all'); ymax = max(Y, [], 'all');
        fprintf("Maximum value found at (%.1f, %1f): %.2f, stride: %d\n", x, y, v, s);
        rectangle('Position',[xmin ymin xmax-xmin ymax-ymin], 'EdgeColor','r');
        plot(x, y, 'wo');
        hold on;
    
        X = X(i-d+1:i+d, j-d+1:j+d);
        Y = Y(i-d+1:i+d, j-d+1:j+d);
        V = V(i-d+1:i+d, j-d+1:j+d);
        
        s = s / 2;
    end
    
    
    
    
    
    
    
    