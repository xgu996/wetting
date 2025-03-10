x = [5, 15, 25, 35, 45];
x_wetting = [5, 15, 25, 35, 45, 55, 65];
y_left = [0.7866, 0.8104, 0.78476, 0.79248, 0.786128];
y_right = [0.7866, 0.79281894, 0.7845156, 0.81039, 0.786018];
y_wetting = [0.7862, 0.78937, 0.764255, 0.76699, 0.74178, 0.744470, 0.71921]; 

figure

% scatter(x, y_left, 100, 'o', 'MarkerEdgeColor','#0072BD',...
%     'MarkerFaceColor',[1 1 1], 'LineWidth',2)
% hold on;
scatter(x, y_right, 100, 'x', 'MarkerEdgeColor','#D95319',...
    'MarkerFaceColor',[1 1 1], 'LineWidth',2)
% hold on;
xlim([0 45])
set(gcf,'unit','centimeters','position',[10 5 30 10])
set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
set(gcf,'color',"white"); 

% 
figure
scatter(x_wetting, y_wetting);

xlim([0 65])
set(gcf,'unit','centimeters','position',[10 0 30 20])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',0.71:0.005:0.79)
% set(gca,'ytick',[],'yticklabel',[])
set(gcf,'color',"white"); 

