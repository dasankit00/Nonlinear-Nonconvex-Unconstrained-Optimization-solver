function out = rosenbrock(x, option)
    %   out = ROSENBROCK(x, option) returns the function val, gradient and Hessian at x
    %   option 0 = displays function value at x
    %   option 1 = displays gradient value at x
    %   option 2 = displays Hessian value at x
    %   x = point of evaluation
  
    n = length(x);
    switch option
        case 0 % Function value
            f = 0;
            for i = 1:n-1
                f = f + 100 * (x(i+1) - x(i)^2)^2 + (1 - x(i))^2;
            end
            out = f;
        
        case 1 % Gradient
            g = zeros(n, 1);
            for i = 1:n-1
                g(i) = g(i) - 400 * x(i) * (x(i+1) - x(i)^2) - 2 * (1 - x(i));
                g(i+1) = g(i+1) + 200 * (x(i+1) - x(i)^2);
            end
            out = g;
           
        case 2 % Hessian 
            H = zeros(n);
            for i = 1:n-1
                H(i,i) = 1200 * x(i)^2 - 400 * x(i+1) + 2;
                H(i,i+1) = -400 * x(i);
                H(i+1,i) = -400 * x(i);
                H(i+1,i+1) = 200;
            end
            out = H;
           
    end
end