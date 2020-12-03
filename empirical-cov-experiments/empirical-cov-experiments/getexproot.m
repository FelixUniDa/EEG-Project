function [ path ] = getexproot( )
%GETEXPROOT Summary of this function goes here
%   Detailed explanation goes here

path = strrep(mfilename('fullpath'), mfilename(), '');

end

