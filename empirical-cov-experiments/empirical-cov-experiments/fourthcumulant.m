function [ K ] = fourthcumulant( X, theta )
%% FOURTHCUMULANT
% Computes the directional fourth cumulant of the data contained in X
% Variables:
%   X: n-by-m matrix where n is the dimension and m is the number of points
% Output:
%   theta: array of angles, 
%   K: array of directional cumulants, corresponding to each direction in
%      theta

[n,m] = size(X);

% theta = combnk(0:0.1:(2*pi), n-1);
% if ~theta
% theta = 0:0.1:(2*pi);
% end

% Will be 2-by-length(theta)
u = cell2mat(arrayfun(@tovector, theta, 'UniformOutput', false));

% Project X onto each direction. Ends up m-by-length(theta)
proj = X'*u;
% Estimate fourth and second moment, then combine to get cumulant
fourth = sum(proj.^4,1)/m;
second = sum(proj.^2,1)/m;

K = fourth - 3*(second.^2);
end