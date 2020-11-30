function [ ] = generatesamples( dim, lowerlimit, upperlimit, step, varargin )
%GENERATESAMPLES Get samples from mathematica
%   Detailed explanation goes here

% Set defaults
verbose = false;
exponents = false;

if ~isempty(varargin)
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
                case 'verbose'
                    verbose = strcmpi(varargin{i+1}, 'true');
                case 'exponents'
                    exponents = true;
                    expcode = varargin{i+1};
                case 'seed'
                    seed_given = true;
                    seed = varargin{i+1};
                otherwise
                    error(['Unknown argument: ''' varargin{i} '''']);
            end;
        end;
    end;
end;

% We'll need this later
thisfolder = strrep(mfilename('fullpath'), mfilename(), '');

% Generate the samples from mathematica

params =  [int2str(dim) ' ' int2str(lowerlimit) ' ' ...
    int2str(upperlimit) ' ' int2str(step)];

if exponents
    params = [params ' "' expcode '" ' num2str(seed)];
end

if ispc
    cmd = ['"C:\Program Files\Wolfram Research\Mathematica\10.1\math" -script "' ...
     thisfolder 'mathematicasamples.w" ' params];
else
    cmd = [thisfolder 'mathematicasamples.w ' params];
end

system(cmd);

end