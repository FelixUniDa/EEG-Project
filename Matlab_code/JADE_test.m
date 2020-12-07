%% Jade Test Script 
% Initialize same signals as in python


rng('default');
seed = rng;    % set seed for reproducible results

n_samples = 2000;
time = 0:8/(n_samples -1):8;

s1 = sin(2 * time);  % Signal 1 : sinusoidal signal
s2 = sign(sin(3 * time));  % Signal 2 : square signal
s3 = sawtooth(2 * pi * time);  % Signal 3: sawtooth signal

S = [s1; s2; s3];

S = S + 0.2 * normrnd(0,3,[3,2000]);  % Add noise

S = S'./std(S');  % Standardize data
% Mix data
A = [[1, 1, 1]; [0.5, 2, 1.0]; [1.5, 1.0, 2.0]];  % Mixing matrix
X = S*A';  % Generate observations

B = jader(X',3);
%%
S_ = B*X';