function s = trust_region_cg(H, g, delta, cgopttol, cgmaxiter)
    % TRUST_REGION_CG Solves trust region subproblem using conjugate gradient.
    n = length(g);
    s = zeros(n, 1);
    r = -g;
    d = r;
    for k = 1:cgmaxiter
        Hd = H * d;
        alpha = (r' * r) / (d' * Hd);
        s_new = s + alpha * d;
        if norm(s_new) > delta || (d' * Hd) <= 0
            % Solve for tau where ||s + tau * d|| = delta
            a = d' * d;
            b = 2 * s' * d;
            c = s' * s - delta^2;
            tau = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);
            s = s + tau * d;
            break;
        end
        s = s_new;
        r_new = r - alpha * Hd;
        if norm(r_new) < cgopttol
            break;
        end
        beta = (r_new' * r_new) / (r' * r);
        d = r_new + beta * d;
        r = r_new;
    end
end