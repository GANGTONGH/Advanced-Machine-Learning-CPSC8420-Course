Y = 5*rand(100,100);
X = 5*rand(100,100);
eta = 0.007;
l_L = [];

for i = [1:1000]
    if rem(i,100) == 0
        i
        L(X,Y)
    end
    l_L = [l_L, L(X,Y)];

    X_1 = X - eta * dLdX(X,Y);
    X = X_1;
    i = i+1;
end

plot(l_L)
xlabel('Iterations')
ylabel('Objective')

%%
function drv = dLdX(X,Y)
    [U,S,V] = svd(X,0);
    drv = (X - Y) + U * V';
end

function obj = L(X,Y)
    obj = 1/2 * norm(X-Y, 'fro')^2 + norm(svd(X,0),1);
end