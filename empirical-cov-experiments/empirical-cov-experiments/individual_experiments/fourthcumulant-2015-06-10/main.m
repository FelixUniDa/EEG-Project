%% Set defaults

warning('off','MATLAB:normest:notconverge');
warning('off','optim:linprog:AlgOptsWillError');

n = 2;
lowerlimit = 1000;
upperlimit = 11000;
step = 4000;
exponents = '{2.1,2.1}';
orthogonalmix = true;
seed = 452;
rng(seed);

algorithm = 'pow3';

numberofruns = 5;
sizes = lowerlimit:step:upperlimit;

addpath('../../');
addpath('../../third_party');

delete('output.txt');
diary('output.txt');
tic
%% --------------------------------------------------------------------

% Generate the samples from mathematica
generatesamples(n, lowerlimit, upperlimit, step, ...
    'exponents', exponents, ...
    'seed', seed);

% Generate random mixing matrix A from standard gaussian
A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
A = A*(inv(diag(rownorm(A'))));
if orthogonalmix
    A = orth(A); % Optional, based on whether we want an orthonormal basis
end

% Set up data to be plotted
amarierrors = zeros(1,length(sizes));
frobeniuserrors = zeros(1,length(sizes));

exp_root = getexproot();

for i = 1:length(sizes)
    %% Get the current sample size
    % Currently the input from mathematica is m-by-n
    S = csvread([exp_root 'samples/sample-' int2str(sizes(i)) '.csv']);

    % X will be n-by-m, column vectors are the samples
    % n is the dimension (number of sensors)
    % m is the number of samples (columns of X)
    X = A * S';
    [n, m] = size(X);
    
    figure();
    set(gcf,'numbertitle','off','name', [num2str(sizes(i)) ' Samples']);
    
    %% Plot with no damping or orthogonalization
    
    subplot(2, 2, 1);
    hold on;
    plot(X(1,:), X(2,:), '.');
    
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(X);
    theta = 0:0.1:(2*pi);
    
    % Plot the fourth cumulant
    polar(theta, rho, '-');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
    
    %% Plot with oracle and damping
    
    orthogonalizer = orth(A) * inv(A);
    Xnew = orthogonalizer * X;
    
    % Find a reasonably good value for R
    Z = unifrnd(0,1,1,size(Xnew,2));
    R = 1;
    Kest = 0;
    % Currently a bad idea to estimate K_{X_R} from the same samples
    % that we're going to use later, but can be fixed easily
    while Kest <= 0.5
        % At termination, we have R large enough and already know the
        % values Exp[-Norm[x]^2/R^2 for each sample point x
        R = R*2;
        threshold = exp(-sum(Xnew.^2,1)/R^2);
        Kest = mean(threshold);
    end

    disp(['Chosen R: ' int2str(R)]);

    Xdamp = damp(Xnew,R);
    
    subplot(2, 2, 2);
    hold on;
    plot(Xdamp(1,:), Xdamp(2,:), '.');
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(Xdamp);
    
    % Plot the fourth cumulant
    polar(theta, rho, '-');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
    
    %% Plot with covariance orthogonalization and damping
    
    orthogonalizer = inv(sqrtm((1/m) * (X * X')));
    Xnew = orthogonalizer * X;
    
    % Find a reasonably good value for R
    Z = unifrnd(0,1,1,size(Xnew,2));
    R = 1;
    Kest = 0;
    % Currently a bad idea to estimate K_{X_R} from the same samples
    % that we're going to use later, but can be fixed easily
    while Kest <= 0.5
        % At termination, we have R large enough and already know the
        % values Exp[-Norm[x]^2/R^2 for each sample point x
        R = R*2;
        threshold = exp(-sum(Xnew.^2,1)/R^2);
        Kest = mean(threshold);
    end

    disp(['Chosen R: ' int2str(R)]);

    Xdamp = damp(Xnew,R);
    
    subplot(2, 2, 3);
    hold on;
    plot(Xdamp(1,:), Xdamp(2,:), '.');
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(Xdamp);
    
    % Plot the fourth cumulant
    polar(theta, rho, '-');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
    
    %% Plot with centroid orthogonalization and damping
    
    orthogonalizer = centroidOrthogonalizer(X);
    Xnew = orthogonalizer * X;
    
    % Find a reasonably good value for R
    Z = unifrnd(0,1,1,size(Xnew,2));
    R = 1;
    Kest = 0;
    % Currently a bad idea to estimate K_{X_R} from the same samples
    % that we're going to use later, but can be fixed easily
    while Kest <= 0.5
        % At termination, we have R large enough and already know the
        % values Exp[-Norm[x]^2/R^2 for each sample point x
        R = R*2;
        threshold = exp(-sum(Xnew.^2,1)/R^2);
        Kest = mean(threshold);
    end

    disp(['Chosen R: ' int2str(R)]);

    Xdamp = damp(Xnew,R);
    
    subplot(2, 2, 4);
    hold on;
    plot(Xdamp(1,:), Xdamp(2,:), '.');
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(Xdamp);
    
    % Plot the fourth cumulant
    polar(theta, rho, '-');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
    
    savefig(num2str(sizes(i)));
end