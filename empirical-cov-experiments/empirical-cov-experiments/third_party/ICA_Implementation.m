% TODO:  Consider the case where there are more dimensions than signals.

function [S, A_inv, b, totalSteps] = ICA_Implementation(X, functionChoice, ...
    preprocessingChoice, algorithm, enforceOrthogonality, epsilon, ...
    maxIterations, alpha, verbosity)
% function [S, A_inv, b, totalSteps] = ICA_Implementation(X, functionChoice, ...
%     preprocessingChoice, algorithm, enforceOrthogonality, epsilon, ...
%     maxIterations, verbosity)
%   Inputs:
%       X -- n x m matrix containing m mixed signals in its columns
%       functionChoice -- A string stating which function (or contrast)
%       will be used to actually run the ICA algorithm.  The options are:
%           'k3'    -- third cumulant, or skew.
%           'k4'    -- fourth cumulant
%           'rk3'   -- Welling's 3rd (outlier) robust cumulant.
%                      Only valid under low Gaussian noise.  Typical usage
%                      would be with standard whitening or an
%                      outlier-robust variant therein (not provided).
%           'rk4'   -- Welling's 4th (outlier) robust cumulant.
%                      Only valid under low Gaussian noise.  Typical usage
%                      would be with standard whitening or an
%                      outlier-robust varient therein (not provided).
%               NOTE:  In numerical experiments, the Welling-robust cumulants
%               require more samples to be accurate in the GR-ICA setting with
%               the implemented forms of whitening than the standard cumulants.
%               However, the preprocessing and update techniques chosen are not
%               particularly favourable to the Welling-robust cumulants, and 
%               were designed with traditional cumulants in mind.
%           'log cosh'-- FastICA contrast sometimes refered to as tanh for
%                        its first derivative.
%           'normal' -- proportional to normal pdf.  Also a FastICA
%                       contrast.
%       function choices are valid with the following algorithms and 
%       preprocessing steps:   Key:  x - valid
%                                    * - valid with caveats (not
%                                        recommended)
%          choice   GI-ICA   whitening    quasi-orthogonalization
%           'k3'      x         x                 x
%           'k4'      x         x                 x
%           'rk3'     x         x                 *
%           'rk4'     x         x                 *
%       The robust cumulants are based on the robust moments and cumulants
%       as defined in Max Welling's paper:  "Robust Higher Order
%       Statistics".  In that paper, they were created with standard
%       whitening in mind.  There is no reason to believe they can be used
%       in the presence of additive Gaussian noise in general, which is
%       precisely the situation that would make the current
%       quasi-orthogonalization routine of interest.
%
%       algorithm -- Determines the choice of algorithm to run:
%           'GI-ICA'  -- Gradient Iteration ICA.  Only valid for
%                        cumulant functions and cumulant-like functions
%                        (i.e., the robust cumulants).
%
%       preprocessingChoice -- A string stating which function shall be
%       used for preprocessing the data to achieve orthogonality.  The
%       choices are:
%           'whiten'    -- Use this for standard whitening and centering
%           'none'      -- Use this if the data is already whitened and
%                          centered.
%           'quasi-orthogonalize'
%                       -- Uses the directional fourth cumulant method for
%                          quasi-orthogonalizing (sometimes called
%                          quasi-whitening) the data.
%
%       enforceOrthogonality
%             -- If set to 1, then after each update, the current guess of
%                the column of R (the rotation matrix to be recovered) is
%                made orthogonal to all previously recovered directions.
%       epsilon -- precision parameter for deflation stopping condition.
%       maxIterations -- maximum number of iterations before timing out
%           finding one column of A_inv in the deflationary step.
%       alpha -- only used for Welling's "robust cumulants."  This is the
%                robustness factor defined in that paper.
%       verbosity -- Flag which determines the level of output generates.
%           0 is the least verbose, and 3 is the most verbose, with integer
%           values within this range being valid.
%   Outputs:
%       S -- d x N matrix containing the N demixed latent signals samples
%            in its columns.  Under Gaussian noise, the noise remains
%            present in the samples under the new linear combination.
%       A_inv -- The demixing matrix.
%       b -- The offset of mean(X, 2) from the origin.
%       totalSteps -- the number of update steps performed during the
%                     iterative portion of the ICA procedure.
%    Note that s = A_inv * (x - b) for any observation x in X gives the
%    demixing formula.

    %% Constants and Flags
    N = size(X, 2);
    d = size(X, 1);
    totalSteps = 0;
        
    %% preprocessing
    if strcmp(preprocessingChoice, 'whiten')
        [X, L, b] = qwhiten(X, 'whiten');
    elseif strcmp(preprocessingChoice, 'quasi-orthogonalize')
        [X, L, b] = qwhiten(X, 'id quas-orth');
    elseif strcmp(preprocessingChoice, 'none')
        L = eye(size(X, 1));
        b = zeros(size(X, 1), 1);
    else
        fprintf(2, 'ERROR ICA_Implementation:  invalid preprocessing choice');
        return
    end
    
    %% Define function required for the iterative update step
    df = selectFunction(functionChoice, alpha);
    
    A_inv = zeros(d, d);
    for iteration=1:d
        orthFlag = 1;  % Always start by enforcing orthogonality
        
        if verbosity >= 3
            fprintf('Searching for component %d:\n', iteration);
        end
        rowsFound = iteration - 1;
        
        %% choose a random starting vector and orthogonalize
        u = normrnd(0, 1, 1, d)';
        u_prev = zeros(d, 1);
        if rowsFound > 0
            u = u - A_inv(1:rowsFound, :)' * A_inv(1:rowsFound, :) * u;
        end
        u = u / norm(u);

        %% Run update algorithm
        steps = 0;
        while ( 1 ) % Loop exit condition in the following if block
            if ( steps >= maxIterations || norm(u_prev - u) <= epsilon || ...
                norm(u_prev + u) <= epsilon )
                if orthFlag && ~enforceOrthogonality && iteration > 1
                    orthFlag = 0; % relax orthogonality constraint.
                    if verbosity >= 3
                        fprintf('Orthogonality constraint relaxed')
                    end
                else
                    break;  % Exit condition
                end
            end
            
            if verbosity >= 3
                fprintf('.');
            end
            u_prev = u;

            if strcmp(algorithm, 'GI-ICA')
                u = df(u, X, N, d);
            else
                error('invalid update step type');
            end

            % enforce orthogonality constraint when desired.
            if orthFlag && iteration > 1
                u = u - A_inv(1:rowsFound, :)' * A_inv(1:rowsFound, :) * u;
            end
            
            u = u / norm(u);
            steps = steps + 1;
        end
        if (verbosity >= 2)
            if verbosity >= 3
                fprintf('\n')
            end
            fprintf('component %d found in %d iteration(s).\n', iteration, steps);
        end
        A_inv(iteration, :) = u;
        totalSteps = totalSteps + steps;
    end    
    
    %% return results
    S = A_inv * X; % Recall that X has already been whitened / preprocessed.
    A_inv = A_inv * L;
end

function [df] = selectFunction(s, alpha)
%   function [f, df, dd_f] = selectFunction(s)
%   input: 
%   s -- a string containing the choice of function to be used as
%        the ICA contrast.  The options are a follows:  
%    'k3', 'm3'  -- third cumulant (which is also the third central
%                   moment).
%    'k4'  -- forth cumulant
%    'rk3' -- 3rd "robust cumulant" (in the sense of Welling's paper)
%    'rk4' -- 4th "robust cumulant" (in the sense of Welling's paper)
%   alpha -- only needs to be specified for robust cumulants.  It is the
%            robustness parameter.
%   
%   output:
%   All outputs are function handles which give ant estimate of the
%   contrast choice over the data.  f refers to the contrast function from
%   which the ICA implementation is derived.  Note that only f's
%   derivatives are used in the update step.
%   df -- gradient of f(u'*x) with respect to u.
    switch lower(s)
        case {'k3', 'm3'}
            % Note:  The third cumulant and third central moment happen to 
            % be the same.  Since we will renormalize and there is only one
            % term, we do not need to worry about the bias.
%             f = @(u, X, N, d) ...
%                 1/N * sum((u'*X).^3, 2);
            df = @(u, X, N, d) ...
                3/N * X*((X'*u).^2);
        case 'k4'
            % Redone to be an unbiased estimator up to normalization
%             f = @(u, X, N, d) ...
%                 ((N+1) * sum((u'*X).^4, 2)/N - 3*(N-1)*(sum((u'*X).^2)/N).^2) ...
%                  * (N^2 / ((N-1)*(N-2)*(N-3)));
            df = @(u, X, N, d) ...
                ((N+1)*4/N * X*((X'*u).^3) ...
                - (N-1)*12/N^2*sum((u'*X).^2, 2)*(X*(X'*u))) ...
                * (N^2 / ((N-1)*(N-2)*(N-3)));
        case 'rk3'
%             f = @(u, X, N, d) rk3(u, X, N, alpha);
            df = @(u, X, N, d) grk3(u, X, N, alpha);
        case 'rk4'        
%             f = @(u, X, N, d) rk4(u, X, N, alpha);
            df = @(u, X, N, d) grk4(u, X, N, alpha);
        otherwise
            error('INVALID FUNCTION CHOICE');
    end
end

function ret = rk3(u, X, N, alpha)
    % Approximates the u-directional Welling's robust third cumulant from
    % data with robustness parameter alpha.
    rm0 = r_m(X, 0, u, alpha, N);
    rm1 = r_m(X, 1, u, alpha, N);
    rm2 = r_m(X, 2, u, alpha, N);
    rm3 = r_m(X, 3, u, alpha, N);

    ret = rm3/rm0 - 3*rm1*rm2/rm0^2 + 2*(rm1/rm0)^3;
end

function ret = grk3(u, X, N, alpha)
    % Approximates the gradient (with respect to u) of the u-directional
    % Welling's robust third cumulant from data with robustness parameter
    % alpha.
    alpha = 2;
    gm1 = gr_m(X, 1, u, alpha, N);
    gm2 = gr_m(X, 2, u, alpha, N);
    gm3 = gr_m(X, 3, u, alpha, N);
    rm0 = r_m(X, 0, u, alpha, N);
    rm1 = r_m(X, 1, u, alpha, N);
    rm2 = r_m(X, 2, u, alpha, N);

    ret = (6*rm1^2*gm1 - 3*rm0*(rm2*gm1+rm1*gm2) + rm0^2*gm3 ) ...
        / rm0^3;
end

function ret = rk4(u, X, N, alpha)
    % Approximates the u-directional Welling's robust fourth cumulant from
    % data with robustness parameter alpha.
    m0 = r_m(X, 0, u, alpha, N);
    m1 = r_m(X, 1, u, alpha, N);
    m2 = r_m(X, 2, u, alpha, N);
    m3 = r_m(X, 3, u, alpha, N);
    m4 = r_m(X, 4, u, alpha, N);
    
    ret = m4/m0 - 3*(m2/m0)^2 - 4*m1*m3/m0^2 - 6*(m1/m0)^4 ...
        + 12*m1^2*m2/m0^3;
end

function ret = grk4(u, X, N, alpha)
    % Approximates the gradient (with respect to u) of the u-directional
    % Welling's robust fourth cumulant from data with robustness parameter
    % alpha.
    gm1 = gr_m(X, 1, u, alpha, N);
    gm2 = gr_m(X, 2, u, alpha, N);
    gm3 = gr_m(X, 3, u, alpha, N);
    gm4 = gr_m(X, 4, u, alpha, N);
    rm0 = r_m(X, 0, u, alpha, N);
    rm1 = r_m(X, 1, u, alpha, N);
    rm2 = r_m(X, 2, u, alpha, N);
    rm3 = r_m(X, 3, u, alpha, N);

    ret = ( rm0^3*gm4 - rm0^2*(6*rm2*gm2 + 4*(rm1*gm3+gm1*rm3)) ...
        + 12*rm0*(2*rm1*rm2*gm1+rm1^2*gm2)- 24*rm1^3*gm1 ) / rm0^4;
end


function muk = r_m(X, k, u, alpha, N)
    % Compute the k^th u-directional robust moment with robust parameter
    % alpha.
    d = size(X, 1);
    muk = 1/N * alpha^(k+d) * ...
        sum((u'* X).^k .* (exp(-1/2*(alpha^2-1) * sum(X.^2, 1))), 2);
end

function gmuk = gr_m(X, k, u, alpha, N)
    % Compute the gradient with respect to u of the k^th 
    % u-directional robust moment with robust parameter alpha.
    d = size(X, 1);
    gmuk = 1/N * k * alpha^(k+d) * ...
        (X* (((X'*u).^(k-1).* (exp(-1/2*(alpha^2-1)*sum(X.^2, 1)))' )));
end

