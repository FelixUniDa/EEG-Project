function [ results, orthdata ] = singlecomparison(dim, lowersize, highsize, step, varargin)
%% SINGLECOMPARISON
% This function will  first generate samples via  call to a mathematica
% script, then run an ICA algorithm.  The mixing matrix A is generated from
% standard gaussians, then the columns are normalized to unit length.
% Inputs:
%  dim: the dimensionality of the samples to be generated
%  lowersize: smallest sample size to be generated
%  highsize: upper bound on sample size
%  step: gap between generated sample sizes
%  varargin: pairs of arguments, first part is a string. details below:
%    'orthogonalmix':       true or false - if true, ensures that the
%                           mixing matrix A is orthogonal
%    'damp':              true or false - if true, performs the dampening
%                           sample rejection
%    'verbose':             true or false - if true, displays various
%                           warnings and messages when relevant
%    'algorithm':           string - selects from the available set of
%                           algorithms currently included:
%                               - 'pow3': FastICA with pow3 nonlinearity
%                               - 'tanh': FastICA with tanh nonlinearity
%                               - 'fpca': Fourier PCA
%                               - 'sobi': SOBI
%                               - 'jade': JADE
%    'regenerate_samples':  true or false - if false, uses the same samples
%                           repeatedly. this can come in handy to avoid the
%                           overhead of invoking mathematica
%    'sanity':              true or false - if true, generates points from
%                           the boolean hypercube instead of mathematica

% We were using n before, so just patch it to use dim for a more readable
% function declaration
n = dim;

%% Set defaults

orthogonalmix = true;
damp = false;
verbose = true;
algorithm = 'pow3';
regenerate_samples = true;
sanity = false;
orthmethod = 'covariance';
exponents = false;
seed = 0;
seed_given = false;
run = 0;
run_given = false;
only_orthogonalize = false;

%% ----------------------- Process Arguments -------------------------

if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        if ~ischar (varargin{i}),
          error (['Unknown type of optional parameter name (parameter' ...
              ' names must be strings).']);
        end
        % change the value of parameter
        switch lower (varargin{i})
            case 'orthogonalmix'
                orthogonalmix = strcmpi(varargin{i+1}, 'true');
            case 'damp'
                damp = strcmpi(varargin{i+1}, 'true');
            case 'verbose'
                verbose = strcmpi(varargin{i+1}, 'true');
            case 'algorithm'
                algorithm = lower(varargin{i+1});
            case 'regenerate_samples'
                regenerate_samples = strcmpi(varargin{i+1}, 'true');
            case 'sanity'
                sanity = strcmpi(varargin{i+1}, 'true');
            case 'orthmethod'
                orthmethod = lower(varargin{i+1});
            case 'exponents'
                exponents = true;
                expcode = varargin{i+1};
            case 'seed'
                seed_given = true;
                seed = varargin{i+1};
            case 'run'
                run_given = true;
                run = varargin{i+1};
            case 'only'
                switch lower(varargin{i+1})
                    case 'orthogonalize'
                        only_orthogonalize = true;
                end
            otherwise
                error(['Unknown argument: ''' varargin{i} '''']);
        end;
    end;
end

%% --------------------------------------------------------------------

% We'll need this later
thisfolder = strrep(mfilename('fullpath'), mfilename(), '');

% Call the sample generation
if regenerate_samples && ~sanity
    if exponents
        generatesamples(dim, lowersize, highsize, step, ...
            'exponents', expcode, ...
            'seed', num2str(seed + run));
    else
        generatesamples(dim, lowersize, highsize, step);
    end
end

% Set up the sample sizes from mathematica
sizes = lowersize:step:highsize;

% Generate random mixing matrix A from standard gaussian
A = mvnrnd(zeros(1,n), eye(n), n);
% Normalize columns of A
A = A*(inv(diag(rownorm(A'))));
if orthogonalmix
    A = orth(A); % Optional, based on whether we want an orthonormal basis
    % A = eye(n);
end
if verbose
    disp(['Smallest angle among columns of A: ' num2str(minangle(A))]);
end

% Set up data to be plotted
amarierrors = zeros(1,length(sizes));
frobeniuserrors = zeros(1,length(sizes));
orthdata = zeros(1,length(sizes));

orthogonalize = ~orthogonalmix && damp;

for i = 1:length(sizes)
    % Currently the input from mathematica is m-by-n
    if sanity
        S = unifrnd(-1, 1, sizes(i),dim);
    else
        S = csvread([thisfolder 'samples/sample-' int2str(sizes(i)) '.csv']);
    end
     
    % X will be n-by-m, column vectors are the samples
    % n is the dimension (number of sensors)
    % m is the number of samples
    X = A * S';
    [n,m] = size(X);

    % Orthogonalize if the mix is non-orthogonal. Assumes mean=0.
    if orthogonalize
        switch orthmethod
            case 'covariance'
                C = (1/m) * (X * X');
                orthogonalizer = inv(sqrtm(C));
            case 'centroid'
                if verbose
                    tic;
                end
                orthogonalizer = centroidOrthogonalizer(X);
                if verbose
                    disp(['Time to orthogonalize via centroid body: ' num2str(toc)]);
                end
            case 'centroidscaling'
                if verbose
                    tic;
                end
                orthogonalizer = centroidOrthogonalizer(X, 'scale');
                if verbose
                    disp(['Time to orthogonalize via centroid body: ' num2str(toc)]);
                end
            case 'oracle'
                % orthogonalizer = orth(A) * inv(A);
                orthogonalizer = orth(A) / A; % Above line, but for speed
            case 'identity'
                orthogonalizer = eye(n);
            otherwise
                error(['Invalid orthogonalizer choice ''' algorithm '''']);
        end
        X = orthogonalizer * X;
        s_min = svds(normalizecols(orthogonalizer*A),1,0);
        if length(s_min) == 1
            orthdata(i) = s_min;
        else
            orthdata(i) = 0;
        end
        disp(['condition number of normalized columns of orthogonalizer*A: '...
            num2str(cond(normalizecols(orthogonalizer*A)))]);
        disp(['s_min of normalized columns of orthogonalizer*A: ' ...
            num2str(s_min)]);
        disp(['s_min of orthogonalizer*A: ' ...
            num2str(svds(orthogonalizer*A,1,0))]);        
        disp(['s_max of orthogonalizer*A: ' ...
            num2str(svds(orthogonalizer*A,1))]);
    else
        s_min = svds(normalizecols(A),1,0);
        if length(s_min) == 1
            orthdata(i) = s_min;
        else
            orthdata(i) = 0;
        end
    end
    
    if ~only_orthogonalize
        % Perform damping, if necessary
        if damp
            C2 = 3;
            R = 1;
            Kest = 0;
            cumest = 0;
            % Currently a bad idea to estimate K_{X_R} from the same samples
            % that we're going to use later, but can be fixed easily
            while Kest <= 0.5 || cumest <= 1/(n^C2)
                % Use a different Z every time
                Z = unifrnd(0,1,1,size(X,2));
                samplecount = size(S,2);
                % At termination, we have R large enough and already know the
                % values Exp[-Norm[x]^2/R^2 for each sample point x
                R = R*2;
                if R == Inf && verbose
                    disp('Could not find large enough R...');
                    disp(['Current Kest: ' num2str(Kest)]);
                    disp(['Current cumest: ' num2str(cumest)]);
                    error('Failed: R too large!');
                end
                Xthreshold = exp(-sum(X.^2,1)/R^2);
                Kest = mean(Xthreshold);
                Sthreshold = exp(-sum(X.^2,1)/R^2);
                tmp =  S(Z <= Sthreshold, :);
                cumest = min(abs(sum(tmp.^4, 1)/size(tmp,1) - 3*(sum(tmp.^2,1)/size(tmp,1))));
                %per = 100*size(tmp,2)/samplecount;
                %if per < 75
                 %   R = R/2;
                %else
                 %   R = R* 2;
            end

            if verbose
                disp(['Chosen R: ' int2str(R)]);
            end

            % Reject samples based on the uniform samples z vs the damping
            firstcount = size(X,2);
            Z = unifrnd(0,1,1,size(X,2));
            X = X(:,Z <= Xthreshold);
            if verbose
                disp(['Samples remaining after rejection: ' ... 
                    int2str(size(X,2)) ' out of ' int2str(firstcount) ...
                    ' (' num2str(100*size(X,2)/firstcount) '%)']);
            end
        end

        % Finally, execute the desired algorithm
        switch algorithm
            case 'pow3'
                % Run ICA with g(u) = u^3
                [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n);
                [whitesig, B, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n, 'only', 'white');

                % Sometimes the FastICA result has less columns that we
                % actually want. Try again, please...
                while size(Aest,2) < dim
                    error('Degenerate!')
                    size(Aest)
                    Aest
                    if verbose
                        disp('Having to correct degeneracy...')
                    end
                    [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n);
                    [~, B, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n, 'only', 'white');
                end
                if ~orthogonalize
                    disp(['Smallest angle among B*A: ' num2str(minangle(B*A))]);
                else
                    disp(['Smallest angle among B*orthogonalizer*A: ' num2str(minangle(B*orthogonalizer*A))]);
                end
                disp(['Whitened center norm:' num2str(norm(mean(whitesig,2)))]);
            case 'tanh'
                % Run ICA with g(u) = tanh(u)
                [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', n, ...
                    'g', 'tanh');

                % Sometimes the FastICA result has less columns that we
                % actually want. Try again, please...
                while size(Aest,2) < dim
                    if verbose
                        disp('Having to correct degeneracy...')
                    end
                    [~, Aest, ~] = fastica(X, 'verbose', 'off', 'numOfIC', ...
                        n, 'g', 'tanh');
                end
            case 'fpca'
                % Run FPCA (naive for now). Note we need to transpose X
                V = phaseCorrect(naiveFPCA(X', 1.6));
                % V = phaseCorrect(recursiveFPCA(eye(size(X)),X', 1.6));
                B = (X*X')/m;
                % Aest = FPCA(X', dim, 1.6);
                % Aest = underdeterminedFPCA(X',10,1.6);
                % size(Aest)
                Aest = sqrtm(B)*V;
            case 'sobi'
                [Aest, ~] = sobi(X);
            case 'jade'
                Aest = inv(jadeR(X));
            case 'simple'
                cov = (X*X')/m;
                [V, ~] = eig(cov);
                Aest = V;
            case 'yeredor'

                [~,m1]=size(X);
                X1 = X(:,1:floor(m1/2));                
                X2 = X(:,floor(m1/2)+1:m1);

                [~,col1] = size(X1);
                [~,col2] = size(X2);
                %disp(size(S2));
                %disp(size(S1));
                M1 = (X1*X1')/col1;
                M2 = (X2*X2')/col2;

                M = M1*inv(M2);
                [V, ~] = eig(M);
                Aest = V;
            otherwise
                error(['Invalid algorithm choice ''' algorithm '''']);
        end

        if orthogonalize
            Aest = inv(orthogonalizer) * Aest;
        end

        % Normalize the results
        Aest = Aest*(inv(diag(rownorm(Aest'))));

        % Use the matching computed by Mukres
        [~, Aestcorrected] = basisEvaluation(A,Aest);

        % Calculate the amari index of \hat{A}^{-1} A
        amarierrors(i) = amari(Aestcorrected\A);
        % Calculate the frobenius difference between \hat{A} and A
        frobeniuserrors(i) = norm(A-Aestcorrected,'fro');
    end
end

% Give the people what they want
results = [
    amarierrors;
    frobeniuserrors
];
end
