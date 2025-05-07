function [alpha, func_evals] = backtracking_line_search(problem, x, d, g, c1)
    % BACKTRACKING_LINE_SEARCH Finds step size satisfying Armijo condition.
    alpha = 1;
    f_x = feval(problem, x, 0);
    func_evals = 1; % To keep track on how many times the fuction is evaluated
    while feval(problem, x + alpha * d, 0) > f_x + c1 * alpha * g' * d
        alpha = alpha / 2;
        func_evals = func_evals + 1;
        if alpha < 1e-10 % To limit the alpha lower bound
            break;
        end
    end
end