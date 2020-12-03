%% Set defaults

dim = 2;
n = dim;
orthogonalmix = true;
dampen = false;
verbose = true;
regenerate_samples = false;
sanity = false;

lowersize=100;
step = 5000;
highsize=15100;
%% --------------------------------------------------------------------

% Generate the samples from mathematica
cmd = ['./mathematicasamples.m ' int2str(n) ' ' int2str(lowersize) ' ' ...
    int2str(highsize) ' ' int2str(step)];
if regenerate_samples && ~sanity
    system(cmd);
end

% Set up the sample sizes from mathematica
sizes = lowersize:step:highsize;

% Generate random mixing matrix A from standard gaussian
A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
A = A*(inv(diag(rownorm(A'))));
if orthogonalmix
    A = orth(A); % Optional, based on whether we want an orthonormal basis
end
if verbose
    disp(['Smallest angle among columns of A: ' num2str(minangle(A))]);
end

% Set up data to be plotted
amarierrors = zeros(1,length(sizes));
frobeniuserrors = zeros(1,length(sizes));

figure()

for i = 1:length(sizes)
    % Currently the input from mathematica is m-by-n
    if sanity
        S = unifrnd(-1, 1, sizes(i),dim);
    else
        S = csvread(['samples/sample-' int2str(sizes(i)) '.csv']);
    end
    
    % Only keep two at a time for this script
    S = S(:,[1,9]);

    % X will be n-by-m, column vectors are the samples
    % n is the dimension (number of sensors)
    % m is the number of samples (columns of X)
    X = A * S';
    X = fastica(X, 'only', 'white');
    [n, m] = size(X);
    
    subplot(2, length(sizes), i);
    hold on;
    plot(X(1,:), X(2,:), '.');
    
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(X);
    theta = 0:0.1:(2*pi);
    
    % Plot the fourth cumulant
    polar(theta, rho, '--');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
    
    % Perform dampening
    Z = unifrnd(0,1,1,size(X,2));
    R = 1;
    Kest = 0;
    % Currently a bad idea to estimate K_{X_R} from the same samples
    % that we're going to use later, but can be fixed easily
    while Kest <= 0.5
        % At termination, we have R large enough and already know the
        % values Exp[-Norm[x]^2/R^2 for each sample point x
        R = R*2;
        threshold = exp(-sum(X.^2,1)/R^2);
        Kest = mean(threshold);
    end

    if verbose
        disp(['Chosen R: ' int2str(R)]);
    end

    % Reject samples based on the uniform samples z vs the damping
    Xdamp = X(:,Z <= threshold);
    if verbose
        disp(['Samples remaining after rejection: ' ... 
            int2str(size(Xdamp,2)) ' out of ' int2str(size(X,2)) ...
            ' (' num2str(100*size(Xdamp,2)/size(X,2)) '%)']);
    end
    
    subplot(2, length(sizes), length(sizes) + i);
    hold on;
    plot(Xdamp(1,:), Xdamp(2,:), '.');
    quiver([0 0], [0, 0], A(1,:), A(2,:));
    
    rho = fourthcumulant(Xdamp);
    
    % Plot the fourth cumulant
    polar(theta, rho, '--');
    ax = gca;
    lim = max(abs([ax.YLim ax.XLim]));
    ax.XLim = [-lim lim];
    ax.YLim = [-lim lim];
    axis square
end