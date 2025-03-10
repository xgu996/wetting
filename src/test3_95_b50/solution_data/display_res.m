mesh_data = load('xyT.mat');

fname = '000_UUU';
solution_data = load([fname, '.mat']);
x = mesh_data.x;
y = mesh_data.y;
T = mesh_data.T;
u = solution_data.u;

% colordef('#993300');

figure;

trisurf(T+1, x, y, u);

set(gcf,'color',"#cccccc"); 

axis([0,1.6,0,0.68]);
axis equal
axis tight
axis off

colormap(parula);
view(2)
shading interp

ax = gca;
ax.Units = 'pixels';
pos = ax.Position;
marg = 0;
rect = [marg, marg, 434, 189];
F = getframe(gca,rect);
ax.Units = 'normalized';

% f=getframe(gca);
imwrite(F.cdata, ['../fig/test3-0_', fname, '.png'],'ResolutionUnit','meter','XResolution',5000);