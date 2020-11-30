function [index] = amari(P)
%% Computes the Amari index of the matrix P

t1 = sum(sum(diag(max(abs(P')))\abs(P), 2) - 1);
t2 = sum(sum(abs(P)/diag(max(abs(P))), 1) - 1);

index = t1 + t2;
end