function H_mod = make_positive_definite(H)
    % MAKE_POSITIVE_DEFINITE Modifies Hessian to ensure positive definiteness.
    xi = 1e-4;
    while min(eig(H)) <= 0
        H = H + xi * eye(size(H, 1));
        xi = 10 * xi;
    end
    H_mod = H;
end