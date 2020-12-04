%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 2;
lowerlimit = 10000;
upperlimit = 10000;
step = 10000;
exponents = '{2.1,2.1}';
orthogonalmix = true;
seed = 352;
rng(seed);

R = 200;

algorithm = 'pow3';

numberofruns = 5;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');

A = eye(n)
% A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
% A = A*(inv(diag(rownorm(A'))));

generatesamples(n, lowerlimit, upperlimit, step, ...
    'exponents', exponents, ...
    'seed', seed);

for i=1:length(sizes)
    % m-by-n
    S = csvread(['../../samples/sample-' num2str(sizes(i)) '.csv']);
    
    % X is n-by-m
    X = A * S';
    [n,m] = size(X);
    
    [~, Aest1, ~] = fastica(X, 'numOfIC', n);
    Aest1 = Aest1*(inv(diag(rownorm(Aest1'))));
    [~, Aest1] = basisEvaluation(A, Aest1)
    
    Xdamp = damp(X,R);
    [~, Aest2, ~] = fastica(Xdamp, 'numOfIC', n);
    Aest2 = Aest2*(inv(diag(rownorm(Aest2'))));
    [~, Aest2] = basisEvaluation(A, Aest2)
    
    orthogonalizer = centroidOrthogonalizer(X);
    Xorthdamp = damp(orthogonalizer * X, R);
    [~, Aest3, ~] = fastica(Xorthdamp, 'numOfIC', n);
    Aest3 = Aest3*(inv(diag(rownorm(Aest3'))));
    [~, Aest3] = basisEvaluation(A, Aest3)
    
    figure();
    hold on;
    %plot(X(1,:),X(2,:), '.');
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    quiver([0 0], [0, 0], Aest1(1,:), Aest1(2,:), 'r');
    quiver([0 0], [0, 0], Aest2(1,:), Aest2(2,:), 'b');
    quiver([0 0], [0, 0], Aest3(1,:), Aest3(2,:), 'g');
    
    rho1 = fourthcumulant(X);
    rho1 = rho1/max(rho1);
    theta = 0:0.1:(2*pi);
    polar(theta, rho1, '-r');
    
    rho2 = fourthcumulant(Xdamp);
    rho2 = rho2/max(rho2);
    theta = 0:0.1:(2*pi);
    polar(theta, rho2, '--b');
    
    rho3 = fourthcumulant(Xorthdamp);
    rho3 = rho3/max(rho3);
    theta = 0:0.1:(2*pi);
    polar(theta, rho3, '-.g');
    
    title(['FastICA - pow3 - R = ' num2str(R)]);
    legend('True A', 'No preprocessing', 'Only damping', 'Centroid', ...
        'Raw', 'Only Damping', 'Centroid');
end

