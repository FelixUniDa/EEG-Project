function [] = myicaplot(amaridata, frobdata, dampenedamaridata, ...
                        dampenedfrobdata, sizes, mytitle)

% Create and display the figure
figure()
set(gcf,'numbertitle','off','name', mytitle)
subplot(2,2,1)
boxplot(amaridata, 'labels', sizes);
title('Raw Data');
xlabel('Sample Size');
ylabel('Amari Index');
ax1 = gca;
subplot(2,2,2)
boxplot(frobdata, 'labels', sizes);
title('Raw Data');
xlabel('Sample Size');
ylabel('Frobenius Error');
ax2 = gca;

% Create and display the figures for dampened data
subplot(2,2,3)
boxplot(dampenedamaridata, 'labels', sizes);
title('Dampened Data');
xlabel('Sample Size');
ylabel('Amari Index');
ax3 = gca;
subplot(2,2,4)
boxplot(dampenedfrobdata, 'labels', sizes);
title('Dampened Data');
xlabel('Sample Size');
ylabel('Frobenius Error');
ax4 = gca;
set(gca,'XTickLabel',sizes)

ax1.YLim(1) = 0;
ax2.YLim(1) = 0;
ax3.YLim(1) = 0;
ax4.YLim(1) = 0;

ax1.YLim(2) = max(ax1.YLim(2), ax3.YLim(2));
ax3.YLim(2) = max(ax1.YLim(2), ax3.YLim(2));

ax2.YLim(2) = max(ax4.YLim(2), ax2.YLim(2));
ax4.YLim(2) = max(ax4.YLim(2), ax2.YLim(2));

if true
    savefig(mytitle);
    print(mytitle, '-dpng');
end

end