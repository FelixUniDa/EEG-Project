function [ Y ] = rownorm( X )
%% ROWNORM computes the norm of every row of a matrix

    Y = sqrt(sum(abs(X) .^ 2, 2));

end

