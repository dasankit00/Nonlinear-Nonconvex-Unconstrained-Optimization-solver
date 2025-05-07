function [alpha, f_evals, g_evals] = wolfe_line_search(problem, x, d, g, f0, c1, c2)
    % WOLFE_LINE_SEARCH Performs a line search satisfying the Wolfe conditions with zooming.
    % Inputs:
    %   problem: Function handle for the problem, returning f for mode=0, g for mode=1
    %   x: Current point (column vector)
    %   d: Search direction (column vector)
    %   g: Gradient at x
    %   f0: Function value at x (passed from optsolver)
    %   c1: Parameter for Armijo condition (e.g., 1e-4)
    %   c2: Parameter for curvature condition (e.g., 0.9), c1 < c2 < 1
    % Outputs:
    %   alpha: Step size satisfying the Wolfe conditions
    %   f_evals: Number of function evaluations
    %   g_evals: Number of gradient evaluations

    % Initialize
    alpha_max = 100;          % Maximum step size
    alpha = 1;                % Initial step size guess
    alpha_prev = 0;           % Previous step size
    slope0 = g' * d;          % phi'(0)
    if slope0 >= 0
        error('Search direction is not a descent direction');
    end
    f_evals = 0;              % Initialize function evaluations counter
    g_evals = 0;              % Initialize gradient evaluations counter
    phi_prev = f0;            % Store phi(alpha_prev)

    % Bracketing phase
    while true
        f_new = feval(problem, x + alpha * d, 0);
        f_evals = f_evals + 1;
        % Check Armijo condition or if function value increases
        if f_new > f0 + c1 * alpha * slope0 || (alpha > 0 && f_new >= phi_prev)
            [alpha, fe, ge] = zoom(alpha_prev, alpha, problem, x, d, f0, slope0, c1, c2);
            f_evals = f_evals + fe;
            g_evals = g_evals + ge;
            return;
        end
        g_new = feval(problem, x + alpha * d, 1);
        g_evals = g_evals + 1;
        slope_new = g_new' * d;
        % Check curvature condition
        if abs(slope_new) <= -c2 * slope0
            return; % Wolfe conditions satisfied
        end
        % If slope becomes non-negative, zoom between alpha and alpha_prev
        if slope_new >= 0
            [alpha, fe, ge] = zoom(alpha, alpha_prev, problem, x, d, f0, slope0, c1, c2);
            f_evals = f_evals + fe;
            g_evals = g_evals + ge;
            return;
        end
        % Update for next iteration
        alpha_prev = alpha;
        phi_prev = f_new;
        alpha = min(2 * alpha, alpha_max); % Double step size, bounded by alpha_max
        if alpha >= alpha_max
            break; % Return current alpha if max reached
        end
    end
end

