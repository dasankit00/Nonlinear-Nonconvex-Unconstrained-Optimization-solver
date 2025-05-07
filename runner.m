function runner(problem, x0)
    % RUNNER Runs all algorithms on a problem with default parameters and tabulates the results.
    algorithms = {'steepestbacktrack', 'steepestwolfe', 'newtonbacktrack', ...
                  'newtonwolfe', 'bfgsbacktrack', 'bfgswolfe', ...
                  'trustregioncg', 'sr1trustregioncg'};

    % Preallocate results structure with 'iterations' added
    results = struct('algorithm', {}, 'iterations', {}, 'convergence', {}, 'objective', {}, ...
                     'grad_norm', {}, 'f_evals', {}, 'g_evals', {}, 'h_evals', {}, 'linsys', {}, 'cpu_time', {});

    for k = 1:length(algorithms)
        fprintf('\nRunning %s\n', algorithms{k});
        [x, info] = optsolver(problem, x0, algorithms{k}, struct());

        % Store results
        results(k).algorithm = algorithms{k};
        results(k).iterations = info.iter;  % Store the number of iterations
        % Convert the final x vector to a string
        str = sprintf('%.4f, ', x);
        str = str(1:end-2); % Remove trailing ', '
        results(k).minimized_x = ['[' str ']'];
        if info.convergence == 1
            results(k).convergence = 'Converged';
        else
            results(k).convergence = 'Not Converged';
        end
        results(k).objective = info.objective;
        results(k).grad_norm = info.grad_norm;
        results(k).f_evals = info.f_evals;
        results(k).g_evals = info.g_evals;
        results(k).h_evals = info.h_evals;
        results(k).linsys = info.linsys;
        results(k).cpu_time = info.cpu_time;
    end

    % Convert to table and display
    T = struct2table(results);
    disp(T);
end