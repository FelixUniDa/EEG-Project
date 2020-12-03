function centroidPlot(X)

theta = 0:0.05:2*pi;
points = length(theta);
%rho = zeros(1,points);
point = zeros(2,points);
% for i =1:points
%     [point(1), point(2)] = pol2cart(theta(i),1);
%     rho(i) = 1/minkowskiCentroidGurobi(X, point);
% end

[point(1,:), point(2,:)] = pol2cart(theta,1);

rho = 1./minkowskiCentroidGurobi(X, point);
figure();
polar(theta, rho);