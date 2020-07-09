%% Load the dVRK scene first
close all; clear; init;
set(0,'defaulttextinterpreter','latex');

bg_path = "./figures/top_down.png";
bg = imread(bg_path);
bg = imrotate(bg, -90);

tl = [0, -0.875];
br = [1.75 0.875];

%% Display the raw scores from image;

% Configurations that are subject to change
data_dir = "workspace_x0.3_0.3_y0.3_0.3_two_arms_ik_2/";
base_dir = base_dir + "cone/";
n = 512; 
[y_arm1, xmin_psm1, xmax_psm1] = loadCombinedScore("psm1", data_dir, base_dir, n);
[y_arm2, xmin_psm2, xmax_psm2] = loadCombinedScore("psm2", data_dir, base_dir, n);

%% Plot figures
nx = 200; ny = 200;
[X_2d_arm1, Y_2d_arm1] = meshgrid(linspace(xmin_psm1(1), xmax_psm1(1), nx), linspace(xmin_psm1(2), xmax_psm1(2), ny));
[X_2d_arm2, Y_2d_arm2] = meshgrid(linspace(xmin_psm2(1), xmax_psm2(1), nx), linspace(xmin_psm2(2), xmax_psm2(2), ny));

% Combined Score
figure();

% Background without overlay;
ax1 = subplot(2,2,1);
imshow(bg);
title("Top-down View");

% Combined score 
levellist = [0.1 0.25 0.5 0.75 0.9 0.95];
cm = 'parula';

% PSM1
ax2 = subplot(2,2,2);
axis off;
img_arm1 = imresize(y_arm1, [nx, ny])';
imagesc([xmin_psm1(1) xmax_psm1(1)], [xmin_psm1(2) xmax_psm1(2)], img_arm1, [0 1]); hold on;
[cont, cont_h] = contour(X_2d_arm1, Y_2d_arm1, img_arm1); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('Y'); 
ylabel('X');
title('Combined Score PSM1');
hold on;
colormap(cm);

% PSM2;
ax3 = subplot(2,2,3);
axis off;
img_arm2 = imresize(y_arm2, [nx, ny])';
imagesc([xmin_psm2(1) xmax_psm2(1)], [xmin_psm2(2) xmax_psm2(2)], img_arm2, [0 1]); hold on;
[cont, cont_h] = contour(X_2d_arm2, Y_2d_arm2, img_arm2); hold on;
cont_h.LevelList = levellist;
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 6);
xlabel('Y'); 
ylabel('X');
title('Combined Score PSM2');
hold on;
colormap(cm);

%% Overlay plots
% Set transparency value of the pixels
figure();

imAlphaData_arm1=0.5 * ones(size(img_arm1));
imAlphaData_arm2=0.5 * ones(size(img_arm2));

% ax3 = subplot(2,2,4);
ibg2 = image(bg);

hold on
%   Overlay the image, and set the transparency previously calculated
iim1 = imagesc(img_arm1, 'CDataMapping','scaled', 'XData',[115 282.5],'YData',[77.5 245]);
set(iim1,'AlphaData',imAlphaData_arm1);
iim2 = imagesc(img_arm2, 'CDataMapping','scaled', 'XData',[115 282.5],'YData',[247.5 417.5]);
set(iim2,'AlphaData',imAlphaData_arm2);
% title('Base Position Scores');

xlabel('X');
ylabel('Y');


set(gca, 'XTick', linspace(1, size(bg, 2), 5), 'XTickLabel', linspace(tl(1), br(1), 5)); % 10 ticks 
set(gca, 'YTick', linspace(1, size(bg, 1), 5), 'YTickLabel', linspace(tl(2), br(2), 5));% 10 ticks
colorbar();
