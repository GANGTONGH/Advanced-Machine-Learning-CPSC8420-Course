clear all;

m = 1000;
n = 999;

y = rand(1,m);
w = rand(n,1);
x = [w; rand]
beta = [w; rand];
lambda = 0.1;
l = [];

%%Grd descent
% t = cputime;
% for i = [1:m]
%     x_i = [rand(n,1); 1];
%     y_i = y(i);
%     beta = beta - dLdb(beta,x_i,y_i) * lambda;
%     obj = L(beta,x_i,y_i);
%     l = [l, obj];
% end
% e = cputime - t;
% disp('GD')
% disp(n)
% disp(e)


%Newtons method

% X = [rand(m,n), ones(m,1)];
% p_vec = zeros(1,m);
% 
% t = cputime;
% for i = [1:m]
%     x_i = X(i,:);
%     y_i = y(i);
%     
%     px_i = pxi(beta, x_i);
%     obj = L(beta,x_i,y_i);
%     grd = x_i * (px_i - y_i);
%     
%     W = zeros(m,m);
%     for i = [1:m]
%        W(i,i)=pxi(beta, X(i,:));
%     end
%     hes = X' * W * X;
%     beta = beta - pinv(hes) * grd';
%     
% %     for i = [1:m]
% %         p_vec(i) = pxi(beta, x_i);
% %     end
% %     grd = X' * (p_vec - y)';
% %     
% %     W = zeros(m,m);
% %     for i = [1:m]
% %        W(i,i)=pxi(beta, X(i,:));
% %     end
% %     hes = X' * W * X;
% %     beta = beta - inv(hes) * grd;
% 
%     l = [l, obj];
% end
% e = cputime - t;
% disp('Newton')
% disp(n)
% disp(e)

%%SGD

t = cputime;
alpha = 0.1;
for i = [1:m]
    y_i = y(i);
    x_i = [rand(n,1); 1];
    px_i = pxi(beta, x_i);
    g = (px_i - y_i) * x_i;
    beta = beta - alpha * g;
    obj = L(beta,x_i,y_i);
    l = [l, obj];
end
e = cputime - t;
disp('SGD')
disp(n)
disp(e)

%% Test

%%
plot(l)
xlabel('Iterations')
ylabel('Objective')
%%
function obj = L(beta,x_i,y_i)
    obj = log(1 + exp(dot(beta' , x_i))) - y_i * dot(beta' , x_i);
end

%%
function grd = dLdb(beta,x_i,y_i)
    grd = exp(dot(beta' , x_i))/(1 + exp(dot(beta' , x_i))) - y_i * x_i;
end

%%
function likelihood = pxi(beta, x_i)
    likelihood = 1/(1+exp(-dot(beta', x_i)));
end