%% Plotting the error as a function of R
%

%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 2;
lowerlimit = 5000;
upperlimit = 5000;
step = 2000;

exponents = {};
% exponents{1} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1}';
% exponents{2} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6}';
% exponents{3} ='{2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,6,6}';
% exponents{1} = '{2.1,2.1}';
% exponents{2} = '{2.1,6}';
exponents{1} = '{1.5,1.5}';


seed = 35;
rng(seed);

Rarr = 10:10:100;

algorithm = 'pow3';

numberofruns = 1;
sizes = lowerlimit:step:upperlimit;

% Decide whether or not to make boxplots
do_boxplot = false;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

%% ------------- Begin Experiment -------------

A = eye(n);

for l=1:length(exponents)
    for i=1:length(sizes)
        for j=1:numberofruns
            for k = 1:length(Rarr)
                generatesamples(n, sizes(i), sizes(i), step, ...
                    'exponents', exponents{l}, ...
                    'seed', seed + j);
                
                % Samples are m-by-n
                S=csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);
                
                X = A*S'; % X will be n-by-m
                
                if n == 2
                    %                 figure();
                    %                 hold on;
                    %                 Xmax = max(sqrt(sum(X.^2,1)));
                    %                 Xplot = X/Xmax;
                    %                 plot(Xplot(1,:), Xplot(2,:), '.');
                    %
                    %                 rho = fourthcumulant(X);
                    %                 rho = rho/max(rho);
                    %                 theta = 0:0.1:(2*pi);
                    %                 polar(theta,rho);
                    
                    figure();
                    hold on;
                    [Xdamp, ~, rejects] = damp(X,Rarr(k));
                    Xdampmax = max(sqrt(sum(Xdamp.^2,1)));
                    Xdampplot = Xdamp/Xdampmax;
                    plot(Xdampplot(1,:), Xdampplot(2,:), '.');
                    plot(rejects(1,:), rejects(2,:), 'x');
                    
                    rho = fourthcumulant(Xdamp);
                    rho = rho/max(rho);
                    theta = 0:0.1:(2*pi);
                    polar(theta,rho);
                    title(['Samples accepted and rejected - ' ...
                            num2str(sizes(i)) ' samples - R ' ...
                            num2str(Rarr(k)) ' Exp ' ...
                            exponents{l}
                        ]);
                end
                
                cov = X*X';
                B = inv(sqrtm(cov));
                
                if false && n == 2
                    figure();
                    hold on;
                    X2 = B*X;
                    plot(X2(1,:), X2(2,:), '.');
                    
                    rho = fourthcumulant(X2);
                    rho = rho/max(rho);
                    theta = 0:0.1:(2*pi);
                    polar(theta,rho);
                    
                    figure();
                    hold on;
                    [Xdamp2, ~, rejects2] = damp(X2, R);
                    size(rejects2,2)
                    plot(Xdamp2(1,:), Xdamp2(2,:), '.');
                    plot(rejects2(1,:), rejects2(2,:), 'x');
                    
                    rho = fourthcumulant(Xdamp2);
                    rho = rho/max(rho);
                    theta = 0:0.1:(2*pi);
                    polar(theta,rho);
                end
            end
        end
    end
end