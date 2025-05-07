function v = genhumps(x,o)

% function v = genhumps(x,o)
%
% Author      : Frank E. Curtis
% Problem     : A multi-dimensional problem with a lot of humps.
%               This problem is from the well-known CUTEr test set.
% Input       : x ~ current iterate
%               o ~ evaluation option
%                     0 ~ function value
%                     1 ~ gradient value
%                     2 ~ Hessian value
% Output      : v ~ function, gradient, or Hessian value

% Switch on o
switch o

  case 0

    % Evaluate function
    v = 0;
    for i = 1:4
      v = v + sin(2*x(i))^2*sin(2*x(i+1))^2 + 0.05*(x(i)^2 + x(i+1)^2);
    end

  case 1
    
    % Evaluate gradient  
    v = [4*sin(2*x(1))*cos(2*x(1))* sin(2*x(2))^2                  + 0.1*x(1);
         4*sin(2*x(2))*cos(2*x(2))*(sin(2*x(1))^2 + sin(2*x(3))^2) + 0.2*x(2);
         4*sin(2*x(3))*cos(2*x(3))*(sin(2*x(2))^2 + sin(2*x(4))^2) + 0.2*x(3);
         4*sin(2*x(4))*cos(2*x(4))*(sin(2*x(3))^2 + sin(2*x(5))^2) + 0.2*x(4);
         4*sin(2*x(5))*cos(2*x(5))* sin(2*x(4))^2                  + 0.1*x(5);];

  case 2
    
    % Evaluate Hessian  
    v      = zeros(5,5);
    v(1,1) =  8* sin(2*x(2))^2*(cos(2*x(1))^2 - sin(2*x(1))^2) + 0.1;
    v(1,2) = 16* sin(2*x(1))*cos(2*x(1))*sin(2*x(2))*cos(2*x(2));
    v(2,2) =  8*(sin(2*x(1))^2 + sin(2*x(3))^2)*(cos(2*x(2))^2 - sin(2*x(2))^2) + 0.2;
    v(2,3) = 16* sin(2*x(2))*cos(2*x(2))*sin(2*x(3))*cos(2*x(3));
    v(3,3) =  8*(sin(2*x(2))^2 + sin(2*x(4))^2)*(cos(2*x(3))^2 - sin(2*x(3))^2) + 0.2;
    v(3,4) = 16* sin(2*x(3))*cos(2*x(3))*sin(2*x(4))*cos(2*x(4));
    v(4,4) =  8*(sin(2*x(3))^2 + sin(2*x(5))^2)*(cos(2*x(4))^2 - sin(2*x(4))^2) + 0.2;
    v(4,5) = 16* sin(2*x(4))*cos(2*x(4))*sin(2*x(5))*cos(2*x(5));
    v(5,5) =  8* sin(2*x(4))^2*(cos(2*x(5))^2 - sin(2*x(5))^2) + 0.1;
    v(2,1) = v(1,2);
    v(3,2) = v(2,3);
    v(4,3) = v(3,4);
    v(5,4) = v(4,5);

end