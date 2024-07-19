clear all;

A = [1, 2, 4; 1, 3, 5; 1, 7, 7; 1, 8, 9];
y = [1; 2; 3; 4];

S = svd(A'*A);

beta = [0;0;0];

s_min = min(S); s_max = max(S); t = 1/s_max;

beta_tmp = 1;

beta_optimal = pinv(A'*A)*A'*y;

iter = 0;
tol = 10^-5;
while (beta_tmp ~= beta) 
    grad_f = A'*(A*beta - y);
    beta_1 = beta - t*grad_f;
    beta_tmp = beta;
    conv_rate = norm(beta_1 - beta_optimal,1)/norm(beta - beta_optimal,1);
    beta = beta_1;
    iter = iter + 1;
    if norm(beta - beta_tmp,2) <= tol; break; end
end

fprintf("n_iter = %d\n",iter)
fprintf("Linear convergence rate by original definition: %f\n", ...
    conv_rate)

fprintf("Linear convergence by 1 - σ_min(A'A)/σ_max(A'A): %f\n", ...
    1 - s_min/s_max)
