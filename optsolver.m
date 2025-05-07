function [x, info] = optsolver(problem, x, algorithm, i)
    % OPTSOLVER Solves an optimization problem using the specified algorithm
    % Inputs:
    %   problem   - Function handle to the problem (e.g., 'rosenbrock'), returns f (mode=0), g (mode=1), or H (mode=2)
    %   x         - Initial guess (vector)
    %   algorithm - String specifying the algorithm: 'steepestbacktrack', 'steepestwolfe', 'newtonbacktrack',
    %               'newtonwolfe', 'bfgsbacktrack', 'bfgswolfe', 'trustregioncg', 'sr1trustregioncg'
    %   i         - Optional structure with algorithm parameters
    % Outputs:
    %   x         - Final solution vector
    %   info      - Structure with convergence info and statistics

    % Initialize parameters with defaults if not provided
    if nargin < 4 || ~isstruct(i)
        i = struct();
    end
    defaults = struct('opttol', 1e-6, 'maxiter', 1000, 'c1ls', 1e-4, ...
                      'c2ls', 0.9, 'c1tr', 0.25, 'c2tr', 0.75, ...
                      'cgopttol', 1e-6, 'cgmaxiter', 10, 'sr1updatetol', 1e-8);
    fields = fieldnames(defaults);
    for k = 1:length(fields)
        if ~isfield(i, fields{k})
            i.(fields{k}) = defaults.(fields{k});
        end
    end

    % Initialize counters and start timing
    counters = struct('f_evals', 0, 'g_evals', 0, 'h_evals', 0, 'linsys', 0);
    tic;

    % Determine problem dimension and evaluate initial point
    n = length(x);
    f = feval(problem, x, 0); % Objective function value
    counters.f_evals = counters.f_evals + 1;
    g = feval(problem, x, 1); % Gradient
    counters.g_evals = counters.g_evals + 1;

    % Initialize iteration counter and display header
    iter = 0;
    fprintf('Iter | Objective | Grad Norm\n');

    % Execute the selected algorithm
    switch lower(algorithm)
        case {'steepestbacktrack', 'steepestwolfe'}
            % Steepest descent with line search
            while norm(g) > i.opttol && iter < i.maxiter
                d = -g; % Steepest descent direction
                if strcmpi(algorithm, 'steepestbacktrack')
                    [alpha, fe] = backtracking_line_search(problem, x, d, g, i.c1ls);
                    counters.f_evals = counters.f_evals + fe;
                else
                    [alpha, fe, ge] = wolfe_line_search(problem, x, d, g, f, i.c1ls, i.c2ls);
                    counters.f_evals = counters.f_evals + fe;
                    counters.g_evals = counters.g_evals + ge;
                end
                x = x + alpha * d; % Update position
                f = feval(problem, x, 0);
                counters.f_evals = counters.f_evals + 1;
                g = feval(problem, x, 1);
                counters.g_evals = counters.g_evals + 1;
                iter = iter + 1;
                fprintf('%4d | %9.4e | %9.4e\n', iter, f, norm(g));
            end
            % Check if the point is a saddle point or a minimum
            H = feval(problem, x, 2);
            counters.h_evals = counters.h_evals + 1;
            eigenvalues = eig(H);
            if all(eigenvalues > 0)
                fprintf('Steepest Descent: The solution is a local minimum.\n');
            elseif any(eigenvalues < 0)
                fprintf('Steepest Descent: The solution is a saddle point.\n');
            else
                fprintf('Steepest Descent: The solution has zero eigenvalues; test inconclusive.\n');
            end

        case {'newtonbacktrack', 'newtonwolfe'}
            % Newton's method with line search
            while norm(g) > i.opttol && iter < i.maxiter
                H = feval(problem, x, 2); % Hessian
                counters.h_evals = counters.h_evals + 1;
                H = make_positive_definite(H); % Ensure positive definiteness
                d = -H \ g; % Newton direction
                counters.linsys = counters.linsys + 1;
                if strcmpi(algorithm, 'newtonbacktrack')
                    [alpha, fe] = backtracking_line_search(problem, x, d, g, i.c1ls);
                    counters.f_evals = counters.f_evals + fe;
                else
                    [alpha, fe, ge] = wolfe_line_search(problem, x, d, g, f, i.c1ls, i.c2ls);
                    counters.f_evals = counters.f_evals + fe;
                    counters.g_evals = counters.g_evals + ge;
                end
                x = x + alpha * d;
                f = feval(problem, x, 0);
                counters.f_evals = counters.f_evals + 1;
                g = feval(problem, x, 1);
                counters.g_evals = counters.g_evals + 1;
                iter = iter + 1;
                fprintf('%4d | %9.4e | %9.4e\n', iter, f, norm(g));
            end
            % Check if the point is a saddle point or a minimum
            H = feval(problem, x, 2);
            counters.h_evals = counters.h_evals + 1;
            eigenvalues = eig(H);
            if all(eigenvalues > 0)
                fprintf('Newton: The solution is a local minimum.\n');
            elseif any(eigenvalues < 0)
                fprintf('Newton: The solution is a saddle point.\n');
            else
                fprintf('Newton: The solution has zero eigenvalues; test inconclusive.\n');
            end

        case {'bfgsbacktrack', 'bfgswolfe'}
            % BFGS quasi-Newton method with line search
            B = eye(n); % Initial Hessian approximation
            while norm(g) > i.opttol && iter < i.maxiter
                d = -B \ g; % Search direction
                counters.linsys = counters.linsys + 1;
                if strcmpi(algorithm, 'bfgsbacktrack')
                    [alpha, fe] = backtracking_line_search(problem, x, d, g, i.c1ls);
                    counters.f_evals = counters.f_evals + fe;
                else
                    [alpha, fe, ge] = wolfe_line_search(problem, x, d, g, f, i.c1ls, i.c2ls);
                    counters.f_evals = counters.f_evals + fe;
                    counters.g_evals = counters.g_evals + ge;
                end
                s = alpha * d; % Step
                x_new = x + s;
                f_new = feval(problem, x_new, 0);
                counters.f_evals = counters.f_evals + 1;
                g_new = feval(problem, x_new, 1);
                counters.g_evals = counters.g_evals + 1;
                y = g_new - g; % Gradient difference
                B = damped_bfgs_update(B, s, y); % Update Hessian approximation
                x = x_new;
                f = f_new;
                g = g_new;
                iter = iter + 1;
                fprintf('%4d | %9.4e | %9.4e\n', iter, f, norm(g));
            end
            % Check if the point is a saddle point or a minimum
            H = feval(problem, x, 2);
            counters.h_evals = counters.h_evals + 1;
            eigenvalues = eig(H);
            if all(eigenvalues > 0)
                fprintf('BFGS: The solution is a local minimum.\n');
            elseif any(eigenvalues < 0)
                fprintf('BFGS: The solution is a saddle point.\n');
            else
                fprintf('BFGS: The solution has zero eigenvalues; test inconclusive.\n');
            end

        case 'trustregioncg'
            % Trust region method with conjugate gradient solver
            delta = 1; % Initial trust region radius
            while norm(g) > i.opttol && iter < i.maxiter
                H = feval(problem, x, 2);
                counters.h_evals = counters.h_evals + 1;
                s = trust_region_cg(H, g, delta, i.cgopttol, i.cgmaxiter);
                counters.linsys = counters.linsys + length(s) - 1; % Approx CG iterations
                f_new = feval(problem, x + s, 0);
                counters.f_evals = counters.f_evals + 1;
                pred_red = -g' * s - 0.5 * s' * H * s; % Predicted reduction
                rho = (f - f_new) / pred_red; % Acceptance ratio
                if rho > i.c1tr
                    x = x + s;
                    f = f_new;
                    g = feval(problem, x, 1);
                    counters.g_evals = counters.g_evals + 1;
                end
                % Adjust trust region radius
                if rho > i.c2tr
                    delta = 2 * delta;
                elseif rho < i.c1tr
                    delta = 0.5 * delta;
                end
                iter = iter + 1;
                fprintf('%4d | %9.4e | %9.4e\n', iter, f, norm(g));
            end
            % Check if the point is a saddle point or a minimum
            H = feval(problem, x, 2);
            counters.h_evals = counters.h_evals + 1;
            eigenvalues = eig(H);
            if all(eigenvalues > 0)
                fprintf('Trust Region CG: The solution is a local minimum.\n');
            elseif any(eigenvalues < 0)
                fprintf('Trust Region CG: The solution is a saddle point.\n');
            else
                fprintf('Trust Region CG: The solution has zero eigenvalues; test inconclusive.\n');
            end

        case 'sr1trustregioncg'
            % SR1 quasi-Newton trust region method
            B = eye(n); % Initial Hessian approximation
            delta = 1;
            while norm(g) > i.opttol && iter < i.maxiter
                s = trust_region_cg(B, g, delta, i.cgopttol, i.cgmaxiter);
                counters.linsys = counters.linsys + length(s) - 1;
                f_new = feval(problem, x + s, 0);
                counters.f_evals = counters.f_evals + 1;
                pred_red = -g' * s - 0.5 * s' * B * s;
                rho = (f - f_new) / pred_red;
                if rho > i.c1tr
                    x_new = x + s;
                    g_new = feval(problem, x_new, 1);
                    counters.g_evals = counters.g_evals + 1;
                    y = g_new - g;
                    r = s' * (y - B * s);
                    % Update B if condition is met
                    if abs(r) >= i.sr1updatetol * norm(s) * norm(y - B * s)
                        B = B + ((y - B * s) * (y - B * s)') / r;
                    end
                    x = x_new;
                    f = f_new;
                    g = g_new;
                end
                if rho > i.c2tr
                    delta = 2 * delta;
                elseif rho < i.c1tr
                    delta = 0.5 * delta;
                end
                iter = iter + 1;
                fprintf('%4d | %9.4e | %9.4e\n', iter, f, norm(g));
            end
            % Check if the point is a saddle point or a minimum
            H = feval(problem, x, 2);
            counters.h_evals = counters.h_evals + 1;
            eigenvalues = eig(H);
            if all(eigenvalues > 0)
                fprintf('SR1 Trust Region CG: The solution is a local minimum.\n');
            elseif any(eigenvalues < 0)
                fprintf('SR1 Trust Region CG: The solution is a saddle point.\n');
            else
                fprintf('SR1 Trust Region CG: The solution has zero eigenvalues; test inconclusive.\n');
            end

        otherwise
            error('Unknown algorithm: %s', algorithm);
    end

    % Display final solution
    str = sprintf('%.4f, ', x);
    str = str(1:end-2);
    fprintf('Optimized x: [%s]\n', str);

    % Populate output info structure
    info.convergence = norm(g) <= i.opttol;
    info.objective = f;
    info.grad_norm = norm(g);
    info.f_evals = counters.f_evals;
    info.g_evals = counters.g_evals;
    info.h_evals = counters.h_evals;
    info.linsys = counters.linsys;
    info.cpu_time = toc;
    info.iter = iter;

    % Report convergence status
    if info.convergence
        fprintf('Algorithm %s converged.\n', algorithm);
    else
        fprintf('%s did not converge. Maximum iterations reached.\n', algorithm);
    end
end