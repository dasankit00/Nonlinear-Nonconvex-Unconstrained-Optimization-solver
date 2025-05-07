function [alpha_star, f_evals, g_evals] = zoom(alpha_lo, alpha_hi, problem, x, d, f0, slope0, c1, c2)
    % ZOOM Narrows down the step size interval to satisfy Wolfe conditions.
    % Inputs:
    %   alpha_lo, alpha_hi: Interval bounds
    %   problem, x, d, f0, slope0, c1, c2: Same as in wolfe_line_search
    % Outputs:
    %   alpha_star: Step size satisfying Wolfe conditions
    %   f_evals: Number of function evaluations
    %   g_evals: Number of gradient evaluations

    f_evals = 0;
    g_evals = 0;
    max_iter = 20; % Prevent infinite loops
    for iter = 1:max_iter
        % Use bisection to select trial step size
        alpha_j = (alpha_lo + alpha_hi) / 2;
        f_j = feval(problem, x + alpha_j * d, 0);
        f_evals = f_evals + 1;
        g_j = feval(problem, x + alpha_j * d, 1);
        g_evals = g_evals + 1;
        slope_j = g_j' * d;
        f_lo = feval(problem, x + alpha_lo * d, 0);
        f_evals = f_evals + 1;
        % Update interval based on conditions
        if f_j > f0 + c1 * alpha_j * slope0 || f_j >= f_lo
            alpha_hi = alpha_j;
        else
            if abs(slope_j) <= -c2 * slope0
                alpha_star = alpha_j;
                return; % Wolfe conditions satisfied
            end
            if slope_j * (alpha_hi - alpha_lo) >= 0
                alpha_hi = alpha_lo;
            end
            alpha_lo = alpha_j;
        end
    end
    % Return last alpha_j if max iterations reached
    alpha_star = alpha_j;
end