clear all;

A = [1, 2, 4; 1, 3, 5; 1, 7, 7; 1, 8, 9];
y = [1; 2; 3; 4];

S = svd(A'*A);

beta = [0;0;0];

s_min = min(S); s_max = max(S); 

beta_tmp = 1;

% Linear convergence rate by original definition

tol = 10^-6;
for lambda = [ 0.1, 1, 10, 100, 200 ]
    %step size
    t = 1/(s_max+lambda);

    beta_optimal = pinv(A'*A+lambda)*A'*y;

    iter = 0;
    while (beta_tmp ~= beta) & (iter <= 100000)
        grad_f = 2*A'*(A*beta - y) + lambda*beta;
        beta_1 = beta - t*grad_f;
        beta_tmp = beta;
        conv_rate = norm(beta_1 - beta_optimal,1)/norm(beta - beta_optimal,1);
        beta = beta_1;
        iter = iter + 1;
        if norm(beta - beta_tmp,2) <= tol; break; end
    end
    fprintf("iter = %d\n", iter)
    fprintf("lambda = %d\n", lambda)
    fprintf("iter = %d\n", iter)
    fprintf("Linear convergence rate by original definition: %d\n", ...
        conv_rate)

        % Linear convergence by 1 - σ_min(A'A)/σ_max(A'A)
    
    fprintf("Linear convergence by 1 - (lambda+σ_min(A'A))/(lambda+σ_max(A'A)): %f\n", ...
        1 - (s_min+lambda)/(s_max+lambda))

end

