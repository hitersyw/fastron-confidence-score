function [] = generate2Dsets(n, root_dir, plot_figure)
    
    d = 2; % change number of links of robot
    L = repmat(Link('d', 0, 'a', 1, 'alpha', 0), d, 1);

    R = SerialLink(L);
    axLim = (sum(R.a) + 1)*[-1 1 -1 1 -1 1];
    R.base = trotx(pi/2)*R.base;
    
    time_delay = 0.05;
    
    R.plotopt = {'view', 'x', 'noshading', 'noname', 'ortho', ...
                'workspace', axLim, 'tile1color', [1,1,1], 'delay', time_delay, 'trail', '', ...
                'noshadow', 'nobase', 'nowrist', ...
                'linkcolor', 'b', 'toolcolor', 'b', ...
                'jointcolor', 0.2*[1 1 1], 'jointdiam', 2, 'jointscale', 1, 'scale', 0.5};

    axis(axLim); axis square; hold on;
    opt = R.plot(zeros(1, d));
    w = 2*opt.scale; % get width of robot arm based on its size in the figure
    colors = [0 0.6 0; 0.5 1 0.9; 0.25 0.25 1; 1 0.6 0.8];

    obs_color = [1 0.3 0.3];
    robot_color = [0 0.5 1];
    cm(1,:) = [1 1 1];
    cm(2,:) = [1 1 1];
    cm(3,:) = obs_color;
    
    ob = 3*[-0.1 -0.1; 0.1 -0.1; 0.1 0.1; -0.1 0.1]; % obs base coordinates;

    offset = sqrt(2) * rand(n, 2); % The maximum dist to origin is 2;

    for k = 1:n
        dir = sprintf("CSpace%d", k);
        mkdir(root_dir, dir);
        obs = {}; % reset obstacle
        obs{1} = ob + offset(k, :);
        
        fill3(obs{1}(:,1), -ones(size(obs{1}(:,1))), obs{1}(:,2), obs_color, 'linewidth', 2);
        title(sprintf("CSpace%d", k));
        
        %% Generate dataset;
        N = 5000;
        q = rand(N, d);
        q = pi*(2*q-1);
        y = ones(size(q,1),1);

        for i=1:size(q,1)
            y(i) = 2*(gjk2Darray(generateArmPolygons(R, q(i,:), w), obs)) - 1;
        end

        data = [q y];
        q_filename = sprintf(root_dir + "/" + dir + "/configs.txt");
        dlmwrite(q_filename, data);

        % TODO: format the text in a json string;
        obs_filename = sprintf(root_dir + "/" + dir + "/obs.txt");
        dlmwrite(obs_filename, obs);
        
        if plot_figure
            % Plot C_free vs. C_obs
            [X2, Y2] = meshgrid(linspace(-pi, pi, 100), linspace(-pi, pi, 100));
            img = size(X2);
            for i = 1:size(X2, 1)
                for j = 1:size(Y2, 2)
                    img(i,j) = 2*(~gjk2Darray(generateArmPolygons(R, [X2(i,j) Y2(i,j)], w), obs)) - 1;
                end
            end

            figure(); axis image; hold on; 
            colormap(winter); 
            imagesc(img);
            xticklabels = pi*(2*linspace(0,1,5)-1);
            yticklabels = xticklabels;
            set(gca, 'XTick', linspace(1, size(img, 1), numel(xticklabels)), ...
                'XTickLabel', xticklabels);
            set(gca, 'YTick', linspace(1, size(img, 2), numel(yticklabels)), ...
                'YTickLabel', yticklabels); set(gca,'YDir','normal')
            xlabel("q1"); ylabel("q2"); 
            title(sprintf("CSpace%d", k)); colorbar
            
            % Scatter plot;
            q_free = q(y==1,:);
            q_obs = q(y==-1,:);

            figure();
            scatter(q_obs(:, 1), q_obs(:, 2), 8, 'r', 'filled'); hold on;
            scatter(q_free(:, 1), q_free(:, 2), 8, 'b', 'filled');
            title(sprintf("CSpace%d", k));
            axis([-pi, pi, -pi, pi]); axis square;
        end
    end
end