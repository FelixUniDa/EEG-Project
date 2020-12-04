function [ M ] = phaseCorrect(A)
%% Follows the procedure in the Fourier PCA paper to remove complex values
% from an estimated mixing matrix

M = A;

for j=1:size(A,2)

    col = zeros(size(A,1));
    max = 0;

    options = 0:0.1:(2*pi);
    
    for k=1:size(options)
        tmpcol = real(exp(1i*options(k))*A(:,j));
        colnorm = norm(tmpcol);
       if colnorm > max
           max = colnorm;
           col = tmpcol;
       end
    end
    
    M(:,j) = col / colnorm;

end

end