clc;
clear;
close all;

%%

filename = 'bigbird_compute_vs_noncompute.tsv';
data = tdfread(filename, '\t');

%%
X = 1:9;
A = cat(1, data.compute', data.non_compute');

map = addcolorplus(313);
num = size(A,1);

idx = linspace(40,55,num);
idx = round(idx);
C = map(idx,:);

%%
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 10;

%%
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%%
GO = barh(X, A', 0.8,'stacked','EdgeColor','k');

%%
GO(1).FaceColor = C(1,:);
GO(2).FaceColor = C(2,:);

YTickLabel= {'1024,32' '1024,64' '1024,128'...
    '3072,32' '3072,64' '3072,128' '4096,32' '4096,64' '4096,128'};

set(gca, 'Box', 'on', ... 
    'XGrid', 'on', 'YGrid', 'off', ...
    'TickDir', 'in', 'TickLength', [.01 .01], ...
    'XMinorTick', 'off', 'YMinorTick', 'off', ...
    'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...
    'XTick',0:20:90,...
    'YTick',1:9,...
    'Xlim' ,[0 90],...
    'Ylim' , [0.2 9.8],...
    'Yticklabel',YTickLabel,...
    'Xticklabel',{0:20:90})

xlabel('Duration(ms)');
hLegend = legend([GO(1),GO(2)], ...
    'Compute', 'Access', ...
    'Location', 'southeast','Orientation','vertical', 'FontSize', 16);
hLegend.ItemTokenSize = [5 5];
legend('boxoff');

set(gca, 'FontSize', 14)
set(gca,'Color',[1 1 1])
