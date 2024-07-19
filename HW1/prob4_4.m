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

l_mseTrain = [];
l_mseTest = [];

lambdas = [0 logspace(-10, 10, 10)]

for lambda = lambdas
    Ntest = N - Ntrain;

    Xtrain = x(1:Ntrain,:); ytrain = y(1:Ntrain);
    %Standardization
    Xtraincent = zscore(Xtrain); 
    Xtest = x(Ntrain+1:end,:); ytest = y(Ntrain+1:end);

    XtrainApp = degexpand(Xtraincent,6,1);
    XtestApp = degexpand(Xtestcent,6,1);
    
    beta_hat = pinv(XtrainApp'*XtrainApp + lambda*eye(79))*XtrainApp'*ytrain;

    ytrainPred = beta_hat' * XtrainApp'
    ytestPred = beta_hat' * XtestApp'

    seTrain = 0;
    for i = 1:Ntrain
        seTrain = (ytrainPred(i) - ytrain(i))^2 + seTrain;
    end
    mseTrain = seTrain/Ntrain;
    l_mseTrain = [l_mseTrain, mseTrain];
    
    seTest = 0;
    for i = 1:Ntest
        seTest = (ytestPred(i) - ytest(i))^2 + seTest;
    end
    mseTest = seTest/Ntest;
    l_mseTest = [l_mseTest, mseTest];
end

l_mseTest
l_mseTrain

%% Plotting
hold off
hold on
lambdas_plt = arrayfun(@(x) log10(x), lambdas)
scatter(lambdas_plt,l_mseTrain,'red','filled')
line(lambdas_plt,l_mseTrain,'Color','red')
xlabel("log(lambda)")
ylabel("MSE")
scatter(lambdas_plt,l_mseTest,'blue','filled')
line(lambdas_plt,l_mseTest,'Color','blue')
legend('Training','','Test','')

%% Reference using built-in ridge() function
% 
% l_mseTrain = [];
% l_mseTest = [];
% 
% for lambda = lambdas
%     B = ridge(ytrain, XtrainApp, lambda)
%     ytrainPred = B' * XtrainApp'
%     ytestPred = B' * XtestApp'
% 
%     seTrain = 0;
%     for i = 1:Ntrain
%         seTrain = (ytrainPred(i) - ytrain(i))^2 + seTrain;
%     end
%     mseTrain = seTrain/Ntrain;
%     l_mseTrain = [l_mseTrain, mseTrain];
%     
%     seTest = 0;
%     for i = 1:Ntest
%         seTest = (ytestPred(i) - ytest(i))^2 + seTest;
%     end
%     mseTest = seTest/Ntest;
%     l_mseTest = [l_mseTest, mseTest];
% end
% 
% hold off
% hold on
% lambdas_plt = arrayfun(@ (x) log(x), lambdas)
% scatter(lambdas_plt,l_mseTrain,'red','filled')
% line(lambdas_plt,l_mseTrain,'Color','red')
% xlabel("Training set size")
% %xticks([1:7])
% %xticklabels([25 50 75 100 150 200 300])
% ylabel("MSE")
% scatter(lambdas_plt,l_mseTest,'blue','filled')
% line(lambdas_plt,l_mseTest,'Color','blue')
% legend('Training','','Test','')

%% 
function xx = degexpand(x, deg, addOnes)
    % Expand input vectors to contain powers of the input features
    % This file is from pmtk3.googlecode.com
    
    [n,m] = size(x);
    if nargin < 3
        addOnes = 0; 
    end
    
    xx = repmat(x, [1 1 deg]);
    degs = repmat(reshape(1:deg, [1 1 deg]), [n m]);
    xx = xx .^ degs;
    xx = reshape(xx, [n, m*deg]);
    
    if addOnes
      xx = [ones(n,1) xx];
    end
end