clear all;

data = load('housing.data');
x = data(:, 1:13);
y = data(:, 14);
[n,d] = size(x);
seed = 2; rand('state',seed); randn('state', seed);
perm = randperm(n); % remove any possible ordering fx
x = x(perm,:); y = y(perm);
N = length(data);

l_mseTrain = [];
l_mseTest = [];

for Ntrain = [25 50 75 100 150 200 300]
    Ntest = N - Ntrain;

    Xtrain = x(1:Ntrain,:); ytrain = y(1:Ntrain);
    %Standardization
    Xtraincent = zscore(Xtrain); 
    Xtest = x(Ntrain+1:end,:); ytest = y(Ntrain+1:end);
    Xtestcent = zscore(Xtest);

    XtrainApp = [ones(Ntrain,1) Xtraincent];
    XtestApp = [ones(Ntest,1) Xtestcent];

    beta_hat = pinv(XtrainApp'*XtrainApp)*XtrainApp'*ytrain;

    ytrainPred = beta_hat' * XtrainApp';
    ytestPred = beta_hat' * XtestApp';

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

% Plotting
hold on
scatter([25 50 75 100 150 200 300],l_mseTrain,'red','filled')
line([25 50 75 100 150 200 300],l_mseTrain,'Color','red')
xlabel("Training set size")
ylabel("MSE")
scatter([25 50 75 100 150 200 300],l_mseTest,'blue','filled')
line([25 50 75 100 150 200 300],l_mseTest,'Color','blue')
legend('Training','','Test','')

