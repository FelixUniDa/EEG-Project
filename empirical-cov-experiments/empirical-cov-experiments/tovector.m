function [ v ] = tovector(theta)
%% TOVECTOR
% Converts a desired angle to a unit (column) vector

v = [ cos(theta) sin(theta) ]';
end