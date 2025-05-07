function B = damped_bfgs_update(B, s, y)
    % DAMPED_BFGS_UPDATE Updates BFGS approximation with damping.
    tau = (s' * y) / (s' * B * s);
    if tau >= 0.2               % 0.2 = bfgs damping tolerance
        phi = 1;
    else
        phi = 0.8 / (1 - tau);
    end
    y_tilde = phi * y + (1 - phi) * B * s;
    B = B - (B * s * s' * B) / (s' * B * s) + (y_tilde * y_tilde') / (s' * y_tilde);
end