clear all;

data = load('housing.data');
x = data(:, 1:13);
y = data(:, 14);
[n,d] = size(x);
seed = 2; rand('state',seed); randn('state', seed);
perm = randperm(n); % remove any possible ordering fx
x = x(perm,:); y = y(perm);
N = length(data)

Ntrain = 300;
Ntest = N - Ntrain;
Xtrain = x(1:Ntrain,:); ytrain = y(1:Ntrain);
Xtest = x(Ntrain+1:end,:); ytest = y(Ntrain+1:end);
Xtraincent = zscore(Xtrain);
Xtestcent = zscore(Xtest);

l_beta = [];

lambdas = [logspace(0,4)];

for lambda = lambdas
    beta_test = lassoAlg(Xtraincent, ytrain, lambda);  
    l_beta = [l_beta beta_test];
end

%Plotting
for i = length(l_beta)
    lambdas_plt = arrayfun(@(x) log10(x), lambdas);
    for i = [1 : length(l_beta(:,1))]
        plot(lambdas_plt,l_beta(i,:),'-*')
        hold on
    end
end
colormap(jet(13))
xlabel("log(lambda)")
ylabel("β")
legends = arrayfun(@(x) strcat('β',string(x)), [1:13])
legend('Parameters', legends)
%
% l_beta(1,:)

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