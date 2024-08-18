n = 3;
m = 4;

lambda = 10;
alpha = 1/2;
gamma = lambda*(1-alpha)/(1+lambda*alpha)

X = rand(m,n);
y = 5 * rand(m,1);

Xcent = zscore(X);

X_1 = (1 + lambda*alpha) * [Xcent; eye(n)];
y_1 = [y; zeros(n,1)];

%% Output L in each step
mat_obj = lassoAlg_step(X_1, y_1, gamma)
l_obj = reshape(mat_obj,1,[])

plot(l_obj)
xlabel('Iterations (on β_i)')
ylabel('Objective')

%%
lassoAlg(X_1, y_1, gamma);
%% Lasso Optimization Algorithm %%
% inputs: A (nxd matrix), y (nx1 vector), lam (scalar)
% return: xh (dx1 vector)

function xh = lassoAlg(A,y,lam)     
    xnew = rand(size(A,2),1);   % "initial guess" 
    xold = xnew+ones(size(xnew)); % used zeros so the while loop initiates
    loss = xnew - xold;
    thresh = 10e-3;     % threshold value for optimization

    while norm(loss) > thresh
        xold = xnew;    % need to store the previous iteration of xh
        for i = 1:length(xnew)
            a = A(:,i);     % get column of A
            p = (norm(a,2))^2;
            % from notes: -t = sum(aj*xj) - y for all j != i
            % i.e., sum(aj*xj) - ai*xi - y (my interpretation)
            % hence t = (above) * -1
            % want to be sure this the correct definition of t?
            t =  a*xnew(i) + y - A*xnew; 
            q = a'*t;
            % update xi
            xnew(i) = (1/p) * sign(q) * max(abs(q)-lam, 0);
        end
        loss = xnew - xold;     % update loss 
    end
    xh = xnew;
end

%% Lasso Optimization Algorithm %%
% inputs: A (nxd matrix), y (nx1 vector), lam (scalar)
% return: xh (dx1 vector)

function xh = lassoAlg_step(A,y,lam)     
    xnew = rand(size(A,2),1);   % "initial guess" 
    xold = xnew+ones(size(xnew)); % used zeros so the while loop initiates
    loss = xnew - xold;
    thresh = 10e-3;     % threshold value for optimization
    xh = [];

    while norm(loss) > thresh
        xold = xnew;    % need to store the previous iteration of xh
        tmp = [];
        for i = 1:length(xnew)
            a = A(:,i);     % get column of A
            p = (norm(a,2))^2;
            % from notes: -t = sum(aj*xj) - y for all j != i
            % i.e., sum(aj*xj) - ai*xi - y (my interpretation)
            % hence t = (above) * -1
            % want to be sure this the correct definition of t?
            t =  a*xnew(i) + y - A*xnew; 
            q = a'*t;
            % update xi
            xnew(i) = (1/p) * sign(q) * max(abs(q)-lam, 0);
            obj = norm(y - A * xnew, 2) + lam*(1-1/2)/(1+lam*1/2)*norm(xnew,1);
            tmp = [tmp, obj];
        end
        loss = xnew - xold;     % update loss 
        %obj = norm(y - A * xnew, 2) + lam*(1-1/2)/(1+lam*1/2)*norm(xnew,1);
        %xh = [xh, obj];
        xh = [xh; tmp];
    end
end