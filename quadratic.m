function v = quadratic(x,o)

% function v = quadratic(x,o)
%
% Author      : Frank E. Curtis
% Problem     : A randomly generated convex quadratic function; the random seed
%               is set so that the results are reproducable.
% Input       : x ~ current iterate
%               o ~ evaluation option
%                     0 ~ function value
%                     1 ~ gradient value
%                     2 ~ Hessian value
% Output      : v ~ function, gradient, or Hessian value

% Set random number generator seeds
rand('seed',0);
randn('seed',0);

% Generate random data
g = randn(10,1);
H = sprandsym(10,0.5,0.5,1);

% Switch on o
switch o

  case 0
    
    % Evaluate function
    v = g'*x + (1/2)*x'*H*x;

  case 1
    
    % Evaluate gradient  
    v = g + H*x;

  case 2
    
    % Evaluate Hessian  
    v = H;


end