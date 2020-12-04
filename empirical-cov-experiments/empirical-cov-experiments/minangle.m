function [theta] = minangle(A)
%% Used to compute the minimum angle between any pair of columns of the
% matrix A

pairs = combnk(1:size(A,2),2);

min = pi;

for i = 1:size(pairs,1)
    col1 = A(:,pairs(i,1));
    col2 = A(:,pairs(i,2));
    
    % Normalize
    col1 = col1/norm(col1);
    col2 = col2/norm(col2);
    
    % Calculate the angle
    ang = acos(dot(col1,col2));
    
    if min > ang
       min =  ang;
    end
end

theta = (min/pi)*180;