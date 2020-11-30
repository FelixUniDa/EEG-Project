function [ S, A_inv, b ] = GIICA( X, preprocessing, varargin )
%function [ S, A_inv ] = GIICA( X, varargin )
%   All users should read at least to the dash line.
%   Required Inputs:
%   X:  The mixed data.  Each column of X represents a sample of data to be
%       demixed via ICA.
%   preprocessing: -- designates the choice of method used in the first
%       step of the 2-step ICA algorithm.
%       Values:
%       'quasi-orthogonalize' -- Uses fourth order cumulant 
%           methods to orthogonalize the latent signals.  This method 
%           is robust to arbitrary, additive Gaussian noise.  However, it
%           requires more samples to do accurately than whitening.
%       'whiten' -- Standard whitening more traditionally used in ICA.
%           This technique should be used on noise-free data, as it
%           requires fewer samples than the quasi-orthogonalization routine
%           to achieve good results.
%       'none' -- Use to forgo the preprocessing step.  Note that the 
%           ICA algorithms implemented are only valid for data which
%           is centered and quasi-orthogonalized (which is a weaker
%           condition than whitening on noise free data).  Only use this
%           option if the data already meets these constraints.
%
%   Return Values:
%   S:  The demixed data (S = A_inv * (X - b)).
%   A:  The demixing matrix recovered under the ICA assumptions.
%   b:  The mean of X which is subtracted during centering.
%
%   Optional Inputs:
%   GIICA should be called in the form:
%       GIICA(X, preprocessing, <op string 1>, <op 1 value>, <op string 2>, <op 2 value>, ... )
%   Where the elipses encompass that any number of options can be used.
%   All options are given default values.  Some example calls are:
%       GIICA(X, 'quasi-orthogonalize');
%       GIICA(X, 'whiten');
%       GIICA(X, 'quasi-orthogonalize', 'contrast', 'k3');
%   The first format is for noisy data (with an unknown, 
%   additive Gaussian noise), and the second format is recommended when
%   the data is believed to be noise-free, or has fewer samples than
%   required to run the 'quasi-orthogonalize' version accurately.  The
%   third format uses an optional input to switch to a third-cumulant
%   skew-based implementation of ICA.  The third cumulant is useful when
%   the latent source signals are not symmetric.
%--------------------------------------------------------------------------
%   The possible optional inputs (and default values) are specified below:
%   Options:
%   'contrast' -- designates the function used during the second step of
%       the ICA algorithm.  This function will be used in the
%       gradient-iteration fashion to recover columns of the mixing matrix
%       under orthogonality assumptions.
%       Values:
%       'k3' -- Third cumulant, or the skew.
%       'k4' (default) -- Fourth cumulant.
%       'rk3' -- Welling's robust third cumulant.  Not recommended when
%           there is an additive Gaussian noise.  Robust in the sense of
%           outliers, but requires more samples than the traditional third
%           cumulant.
%       'rk4' -- Welling's robust fourth cumulant.  Not recommended when
%           there is an additive Gaussian noise.  Robust in the sense of
%           outliers, but requires more samples than the traditional fourth
%           cumulant.
%   'robustness factor' -- Used to specify the robustness constant alpha 
%       used by Welling's robust cumulants.
%       Value:  Provide a decimal value.  Default is 2.  Valid values range
%           from 1 (mimicks traditional cumulants) to infinity.
%       example usage:  GIICA(X, 'whiten', 'contrast', 'rk4', 'robustness factor', 1.5)
%   'max iterations' -- Specifies the cap on the number of iterations which
%       can be used to find a single column of A.
%       Value:  Integer value.  Default is 1000.
%       Example usage:  GIICA(X, 'quasi-orthogonalize', 'max iterations', 100)
%   'orthogonal deflation' -- Determines in the second step of ICA (the
%       deflationary recovery of the demixing matrix under an orthogonality
%       constraint) whether orthogonality is enforced between the column of
%       A currently being recovered and all previously recovered columns.
%       The algorithms employed are in theory self-orthogonalizing, but
%       this is not necessarily true on sample data or when the ICA
%       assumptions do not fully hold.
%       Value: 
%       'false' -- the orthogonality constraint is relaxed after
%           convergence is achieved in the orthogonal subspace.  The
%           maximum iterations bounds the sum of orthogonality enforced and
%           orthogonality relaxed iterations.  If convergence is never
%           achieved in the orthogonal subspace, then orthogonality remains
%           enforced despite this flag.
%       'true' (default)
%   'precision'
%       Value:  A decimal value.  During the deflationary gradient
%           gradient iteration step, this value will be used by the
%           stopping criterion.  In particular, if 2 subsequent estimates
%           for a column of the mixing matrix differ (in terms of cosine)
%           by less than the specified precision, the routine finishes.
%           This value must be less than 1.  The default is 0.0001
%       example call:  GIICA(X, 'quasi-orthogonalize', 'precision', 1e-6)
%    'verbose' -- An integer value specifying the level of verbosity.
%       Value:
%       0  -- Displays runtime errors and MatLab generated Warnings only
%           (not recommended).
%       1  -- Displays runtime errors and warnings.
%       2 (default) -- Displays runtime errors, warnings, and high level 
%           progress indicators.
%       3  -- Displays runtime errors, warnings, and gives lower level
%           progress indicators.
%       Example usage:  GIICA(X, 'whiten', 'verbose', 1)
%
%   NOTE:  Where numerical values are expected for the input, we make no
%   guarantees that the error checking that the input is valid is fully
%   carried out.
%
%   NOTE:  The primary references that this work is based on is:
%   Fast Algorithms for Gaussian Noise Invariant Independent Component
%   Analysis.  by:  James Voss, Luis Rademacher, and Mikhail Belkin.
%   (to appear in NIPS 2013)
    
    if (mod(nargin, 2) ~= 0) || (nargin < 2)
        s = sprintf('%s\n%s\n%s', ...
            'Error:  Invalid number of arguments', ...
            'BASIC USAGE:  [S, A_inv, b] = GIICA(X, preprocessing)', ...
            'ADVANCED USAGE:  [S, A_inv, b] = GIICA(X, preprocessing, <op string 1>, <op 1 val>, <op string 2>, <op 2 val>' );
        error(s);
    end
        
    %% Set default parameter values
    alpha = 2; % robustness factor
    contrast = 'k4';
    epsilon = 1e-4; % precision
    orthogonality = int32(1); % orthogonal deflation flag
    maxIterations = int32(1000);
    verbosity = int32(2);

    %% Error checking for preprocessing choice
    preprocessing = lower(preprocessing);
    if ~( strcmp(preprocessing, 'none') || ...
            strcmp(preprocessing, 'quasi-orthogonalize') || ...
            strcmp(preprocessing, 'whiten') )
        error('Usage:  Invalid preprocessing choice:  %s', opVal);
    end
    
    %% Switch parameter values based on user input arguments
    numUserOps = (nargin - 1) / 2;
    for i = 0:numUserOps-1
        option = varargin(2*i + 1);
        option = option{1};
        opVal = varargin(2*i + 2);
        opVal = opVal{1};
        switch option
            case 'contrast'
                opVal = lower(opVal);
                if strcmp(opVal, 'k3') || strcmp(opVal, 'k4') || ...
                        strcmp(opVal, 'rk3') || strcmp(opVal, 'rk4')
                    contrast = opVal;
                else
                    error('Usage:  Invalid contrast function choice:  %s', opVal);
                end
            case 'robustness factor'
                if ~isnumeric(opVal)
                    error('Usage:  robustness factor must have a numerical type.');
                end
                if opVal >= 1
                    alpha = opVal;
                else
                    error('Usage:  robustness factor must be at least 1');
                end
            case 'max iterations'
                if ~isnumeric(opVal)
                    error('Usage:  Max Iterations must have a numerical type.');
                end
                opVal = int32(floor(opVal));
                if opVal <= 0
                    error('Usage:  Max Iterations must be strictly positive.');
                else
                    maxIterations = opVal;
                end
            case 'orthogonal deflation'
                opVal = lower(opVal);
                if strcmp(opVal, 'true')
                    orthogonality = 1;
                elseif strcmp(opVal, 'false')
                    orthogonality = 0;
                else
                    error('Usage:  Invalid choice for orthogonality constraint:  %s', opVal );
                end                
            case 'precision'
                if ~isnumeric(opVal)
                    error('Usage:  precision value must have a numerical type.');
                end
                if opVal < 1
                    epsilon = opVal;
                else
                    error('Usage:  Invalid precision value (must be less than 1):  %f', opVal);
                end
            case 'verbose'
                if ~isnumeric(opVal)
                    error('Usage:  verbosity value must have a numerical type.');
                end
                opVal = int32(floor(opVal));
                if opVal < 0 || opVal > 3
                    error('Usage:  Invalid parameter for ''verbose'' options');
                end
                verbosity = opVal;
            otherwise
                error('Usage:  Invalid option choice:  %s', option);
        end
    end

    %% Run the ICA algorithm given user specificaitons.
    [S, A_inv, b, totSteps] = ...
        ICA_Implementation(X, contrast, preprocessing, 'GI-ICA', ...
            orthogonality, epsilon, maxIterations, alpha, verbosity);
end

