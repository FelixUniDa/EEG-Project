function [ Xdamp, rate, Xrejected ] = damp( X, R )
%DAMP Perform damping on data
%   Detailed explanation goes here

Z = unifrnd(0,1,1,size(X,2));
threshold = exp(-sum(X.^2,1)/R^2);

% Reject samples based on the uniform samples z vs the damping
Xdamp = X(:,Z <= threshold);
Xrejected = X(:,Z > threshold);

rate = size(Xdamp,2)/size(X,2);

if false
    disp(['Samples remaining after rejection: ' ... 
        int2str(size(Xdamp,2)) ' out of ' int2str(size(X,2)) ...
        ' (' num2str(rate) '%)']);
end
end
