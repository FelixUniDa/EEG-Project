function [W , flg] = PowerICA(X, nonlin, W0, mode)
%
% This function is implemented based on Algorithm 1 in paper below.
%   S. Basiri, E. Ollila and V. Koivunen, "Alternative Derivation of 
%   FastICA With Novel Power Iteration Algorithm," in IEEE Signal 
%   Processing Letters, vol. 24, no. 9, pp. 1378-1382, Sept. 2017.
% If you use this function in your publication please cite our paper using 
% the above citation info.
%
% Matlab Code by Shahab Basiri, Aalto University 2017
% shahab.basiri@aalto.fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUT:
%   X: d*n array of mixture recordings (X should be centered and whitened).
%   nonlin: ICA nonlinearities. 
%           It can be either a single string or a d*1 array of strings. 
%           The following nonlinearities are supported.
%           tanh, pow3, gaus, skew, rt06, lt06, bt00, bt02, bt06, bt10, 
%           bt12, bt14, bt16, tan1, tan2, tan3, tan4, gau1, gau2, gau3.
%   W0: the initial start of the algorithm (an orthogonal d*d matrix).
%   mode: can be set either to 'serial' or 'parallel'. The 'serial' mode 
%         is used when only one computing node is available or the dataset
%         is of small size. The 'parallel' mode runs two Matlab instances
%         on different CPU cores. It is recommended for large datasets.
%% OUTPUT:
%   W: PowerICA estimate of demixing matrix (d*d).
%   flg: 1 when the algorithm has converged successfully  and 0 when the
%   algorithm has failed to converge.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[d, n] = size(X);
I = eye(d);
C = X*X'./n - I;
W = zeros(d);
%% Default settings
switch nargin
    case 3
        mode = 'serial';
    case 2
        mode = 'serial'; W0 = orth(randn(d,d)); 
    case 1
        mode = 'serial'; W0 = orth(randn(d,d)); nonlin = 'tanh';
end
%%
if d>n, error('Data should be a d × n array with d < n!');end
if isreal(X)==0, error('Data is not real!');end
if max(abs(C(:)))>1e-10, error('Data should be whitened!');end
if max(abs((mean(X,2))))>1e-10, error('Data should be centered!');end
if size(nonlin,1)==1,nonlin = repmat({nonlin},d,1); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Serial mode
if strcmp(mode,'serial') 
    for k = 1:d-1  
        %% (1) initialize 
        w0 = W0(k,:)';
        %% (2) compute the orthogonal operator
        Orth = (I - W'*W); 
        %% (3-6) compute Node:1 and Node:2 in series
        [w1, gamma1, flg1] = Node1(X, nonlin(k), w0, Orth); 
        [w2, gamma2, flg2] = Node2(X, nonlin(k), w0, Orth);
        flg = flg1*flg2;
        %% (7) choose the demixing vector with larger gamma
        if flg == 1
            if gamma1 > gamma2
                W(k,:) = w1';
            else
                W(k,:) = w2';
            end
        else
            W(k:d,:) = [];
            flg = 0;
            break;
        end
    end
    %% (8) compute the last demixing vector
    if flg == 1, W(d,:) = W0(d,:)*(I - W'*W)/norm(W0(d,:)*(I - W'*W));end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% Parallel mode
elseif strcmp(mode,'parallel') 
    save X; save nonlin; save W0; retries = 100;
    %% run Node:2 on a parallel Matlab instance 
    !matlab -r parnode & 
    for k = 1:d-1
        %% (1) initialize
        w0 = W0(k,:)';
        %% (2) compute the orthogonal operator
        Orth = (I - W'*W);
        save Orth; server('done', 3000, retries) % send Orth to Node:2
        %% (3-6) compute only Node:1 on this Matlab instance 
        [w1, gamma1, flg1] = Node1(X, nonlin(k), w0, Orth);
        %% import the results from Node:2
        client('localhost', 3000,retries) 
        load w2; load gamma2; load flg2;
        flg = flg1*flg2;  %#ok<NODEF>
        %% (7) choose the demixing vector with larger gamma
        if flg == 1
            if gamma1 > gamma2  %#ok<NODEF>
                W(k,:) = w1';
            else
                W(k,:) = w2';  %#ok<NODEF>
            end
            clear w1 gamma1 flg1;
            clear w2 gamma2 flg2;
        else
            W(k:d,:) = [];
            flg = 0;
            break;
        end
    end
    %% (8) compute the last demixing vector
    if flg == 1, W(d,:) = W0(d,:)*(I - W'*W)/norm(W0(d,:)*(I - W'*W));end
    % clean up the disc;
    delete X.mat nonlin.mat W0.mat Orth.mat w2.mat gamma2.mat flg2.mat;   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, gamma, flg] = Node1(X, nonlin, w0,Orth)
[~, n] = size(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxIter = 10000;
epsilon = 0.0001;
flg = 1;
i = 1;
w = w0;
while i <= MaxIter
    wOld = w;
    s = w'*X;
    gs = g(s,nonlin);
    w = X*gs'./n; %(4)
    w = Orth*w; %(5)
    w = w/norm(w); %(6)   
    if norm(w - wOld) < epsilon || norm(w + wOld) < epsilon
        break;
    end
    i = i + 1; %(3)
end
%     fprintf('IC converged after %d iterations\n',i);
if i <= MaxIter
    beta = dg(s,nonlin);
    gamma = abs(s*gs'./n - beta);
else
    %     fprintf('IC did not converged after %d iterations\n',MaxIter);
    w = [];
    flg = 0;
    gamma = -1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, gamma, flg] = Node2(X, nonlin, w0,Orth)
[~, n] = size(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxIter = 10000;
epsilon = 0.0001;
flg = 1;
i = 1;
w = w0;
% Compute the upper bound
s_max = sqrt(sum(X.^2));
gs_max = g(s_max,nonlin);
c = (s_max*gs_max')/n + 0.5;
while i <= MaxIter
    wOld = w;
    s = w'*X;
    gs = g(s,nonlin);
    m = X*gs'./n;
    w = m - c*w;    %(4)
    w = Orth*w;     %(5)
    w = w/norm(w);  %(6)
    if norm(w - wOld) < epsilon || norm(w + wOld) < epsilon
        break;
    end
    i = i + 1;  %(3)
end
%     fprintf('IC converged after %d iterations\n',i);
if i <= MaxIter
    beta = dg(s,nonlin);
    gamma = abs(s*gs'./n - beta);
else
    %     fprintf('IC did not converged after %d iterations\n',MaxIter);
    w = [];
    flg = 0;
    gamma = -1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gs = g(s,nonlin)
% This function computes the ICA nonlinearity for a given input s = w^t*x.
g = nonlin;
if strcmp(g,'tanh')
    g = @(x) tanh(x);
elseif strcmp(g,'pow3')
    g = @(x) x.^3;
elseif strcmp(g,'gaus')
    g = @(x) x .* exp(- (1/2)*x.^2);
elseif strcmp(g,'skew')
    g = @(x) x.^2;
elseif strcmp(g,'rt06')
    g = @(x) max(0,x-0.6).^2;
elseif strcmp(g,'lt06')
    g = @(x) min(0,x+0.6).^2;
elseif strcmp(g,'bt00')
    g = @(x) max(0,x).^2 - min(0,x).^2;
elseif strcmp(g,'bt02')
    g = @(x) max(0,x-0.2).^2 - min(0,x+0.2).^2;
elseif strcmp(g,'bt06')
    g = @(x) max(0,x-0.6).^2 - min(0,x+0.6).^2;
elseif strcmp(g,'bt10')
    g = @(x) max(0,x-1.0).^2 - min(0,x+1.0).^2;
elseif strcmp(g,'bt12')
    g = @(x) max(0,x-1.2).^2 - min(0,x+1.2).^2;
elseif strcmp(g,'bt14')
    g = @(x) max(0,x-1.4).^2 - min(0,x+1.4).^2;
elseif strcmp(g,'bt16')
    g = @(x) max(0,x-1.6).^2 - min(0,x+1.6).^2;
elseif strcmp(g,'tan1')
    g = @(x) tanh(1.25*x);
elseif strcmp(g,'tan2')
    g = @(x) tanh(1.5*x);
elseif strcmp(g,'tan3')
    g = @(x) tanh(1.75*x);
elseif strcmp(g,'tan4')
    g = @(x) tanh(2*x);
elseif strcmp(g,'gau1')
    g = @(x) x .* exp(- (1.07/2)*x.^2);
elseif strcmp(g,'gau2')
    g = @(x) x .* exp(- (1.15/2)*x.^2);
elseif strcmp(g,'gau3')
    g = @(x) x .* exp(- (0.95/2)*x.^2);
else error('Invalid nonlinearity');
end
gs = g(s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Edgs = dg(s,nonlin)
% This function computes E[g'(w^t*x)]for a given input s = w^t*x.
g = nonlin;
if strcmp(g,'tanh')
    dg = @(x) 1-tanh(x).^2;
elseif strcmp(g,'pow3')
    dg = @(x) 3*x.^2;
elseif strcmp(g,'gaus')
    dg =  @(x) (1 - x.^2) .* exp(- (1/2)*x.^2);
elseif strcmp(g,'skew')
    dg =  @(x) 0 ;
elseif strcmp(g,'rt06')
    dg = @(x) 2*max(0,x-0.6);
elseif strcmp(g,'lt06')
    dg = @(x) 2*min(0,x+0.6);
elseif strcmp(g,'bt00')
    dg = @(x) 2*max(0,x) - 2*min(0,x);
elseif strcmp(g,'bt02')
    dg = @(x) 2*max(0,x-0.2) - 2*min(0,x+0.2);
elseif strcmp(g,'bt06')
    dg = @(x) 2*max(0,x-0.6) - 2*min(0,x+0.6);
elseif strcmp(g,'bt10')
    dg = @(x) 2*max(0,x-1.0) - 2*min(0,x+1.0);
elseif strcmp(g,'bt12')
    dg = @(x) 2*max(0,x-1.2) - 2*min(0,x+1.2);
elseif strcmp(g,'bt14')
    dg = @(x) 2*max(0,x-1.4) - 2*min(0,x+1.4);
elseif strcmp(g,'bt16')
    dg = @(x) 2*max(0,x-1.6) - 2*min(0,x+1.6);
elseif strcmp(g,'tan1')
    dg = @(x) 1.25*(1-tanh(1.25*x).^2);
elseif strcmp(g,'tan2')
    dg = @(x) 1.5*(1-tanh(1.5*x).^2);
elseif strcmp(g,'tan3')
    dg = @(x) 1.75*(1-tanh(1.75*x).^2);
elseif strcmp(g,'tan4')
    dg = @(x) 2*(1-tanh(2*x).^2);
elseif strcmp(g,'gau1')
    dg =  @(x) (1 - 1.07*x.^2) .* exp(- (1.07/2)*x.^2);
elseif strcmp(g,'gau2')
    dg =  @(x) (1 - 1.15*x.^2) .* exp(- (1.15/2)*x.^2);
elseif strcmp(g,'gau3')
    dg =  @(x) (1 - 0.95*x.^2) .* exp(- (0.95/2)*x.^2);
end
Edgs = mean(dg(s));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function message = client(host, port, number_of_retries)
% CLIENT read a message over the specified port
% Code by Rodney Thomson.
% see
% http://iheartmatlab.blogspot.fi/2008/08/
% tcpip-socket-communications-in-matlab.html
% for more details
import java.net.Socket
import java.io.*

if nargin == 2
    number_of_retries = 20; % set to -1 for infinite
end

retry        = 0;
input_socket = [];
message      = [];

while true
    
    retry = retry + 1;
    if ((number_of_retries > 0) && (retry > number_of_retries))
        fprintf(1, 'Too many retries\n');
        break;
    end
    
    try
        fprintf(1, 'Retry %d connecting to %s:%d\n', ...
            retry, host, port);
        
        % throws if unable to connect
        input_socket = Socket(host, port);
        
        % get a buffered data input stream from the socket
        input_stream   = input_socket.getInputStream;
        d_input_stream = DataInputStream(input_stream);
        
        fprintf(1, 'Connected to server\n');
        
        % read data from the socket - wait a short time first
        pause(0.5);
        bytes_available = input_stream.available;
        fprintf(1, 'Reading %d bytes\n', bytes_available);
        
        message = zeros(1, bytes_available, 'uint8');
        for i = 1:bytes_available
            message(i) = d_input_stream.readByte;
        end
        
        message = char(message);
        
        % cleanup
        input_socket.close;
        break;
        
    catch
        if ~isempty(input_socket)
            input_socket.close;
        end
        
        % pause before retrying
        pause(1);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function server(message, output_port, number_of_retries)
% SERVER Write a message over the specified port
% Usage - server(message, output_port, number_of_retries)
% Code by Rodney Thomson.
% see
% http://iheartmatlab.blogspot.fi/2008/08/
% tcpip-socket-communications-in-matlab.html
% for more details
import java.net.ServerSocket
import java.io.*

if nargin == 2
    number_of_retries = 20; % set to -1 for infinite
end
retry             = 0;

server_socket  = [];
output_socket  = [];

while true
    
    retry = retry + 1;
    
    try
        if ((number_of_retries > 0) && (retry > number_of_retries))
            fprintf(1, 'Too many retries\n');
            break;
        end
        
        fprintf(1, ['Try %d waiting for client to connect to this ' ...
            'host on port : %d\n'], retry, output_port);
        
        % wait for 1 second for client to connect server socket
        server_socket = ServerSocket(output_port);
        server_socket.setSoTimeout(1000);
        
        output_socket = server_socket.accept;
        
        fprintf(1, 'Client connected\n');
        
        output_stream   = output_socket.getOutputStream;
        d_output_stream = DataOutputStream(output_stream);
        
        % output the data over the DataOutputStream
        % Convert to stream of bytes
        fprintf(1, 'Writing %d bytes\n', length(message));
        d_output_stream.writeBytes(char(message));
        d_output_stream.flush;
        
        % clean up
        server_socket.close;
        output_socket.close;
        break;
        
    catch
        if ~isempty(server_socket)
            server_socket.close
        end
        
        if ~isempty(output_socket)
            output_socket.close
        end
        
        % pause before retrying
        pause(1);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

