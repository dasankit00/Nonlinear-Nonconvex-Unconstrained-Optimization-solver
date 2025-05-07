function v = leastsquares(x,option)

% function v = leastsquares(x,o)
%
% Author      : Frank E. Curtis
% Problem     : A randomly generated nonlinear least squares problem for fitting
%               data points to a bell-shaped curve; the random seed is set so
%               that the results are reproducable.
%               Note: The second derivatives coded are not exact!  However, your
%               code should treat them as if they are exact.
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
z = [2;1;-5;4];
t = 10*rand(100,1); t = sort(t);
y = 0.05*randn(100,1)+z(1)+z(2)*exp(-((t+z(3)).^2)./z(4));

% Evaluate residual
r = x(1)+x(2)*exp(-((t+x(3)).^2)./x(4))-y;

% Evaluate residual Jacobian
if option > 0, J = [     ones(1,100);
                    exp(-((t'+x(3)).^2)./x(4));
               x(2)*exp(-((t'+x(3)).^2)./x(4)).*(-2*(t'+x(3))./x(4));
               x(2)*exp(-((t'+x(3)).^2)./x(4)).*(((t'+x(3)).^2)./(x(4)^2))]; end;

% Switch on o
switch option

  case 0
    
    % Evaluate function  
    v = (1/2)*(r'*r);

  case 1
    
    % Evaluate gradient  
    v = J*r;

  case 2
    
    % Evaluate Hessian  
    v = J*J';

end