function [B, te_acc, tr_acc, obj]=log_reg(y, X, te_y, te_X, lambda, c)
    
    if ~exist('lambda', 'var')
        lambda=0;
    end

    if ~exist('c', 'var')
        c=1e-4;
    end
    
    % set up maximum iteration as stopping criteria
    maxiter=5000;
    K=length(unique(y));
    p=size(X,2);
    B=zeros(p,K-1);
     
    te_acc=zeros(maxiter,1);
    tr_acc=zeros(maxiter,1);
    obj=zeros(maxiter,1);
 
    iter=1;
    
    while (iter <= maxiter)
        
        %compute gradient 
        G=log_grad(y,X,B)-lambda*B;
        
        %gradient ascent 
        B=B+c*G;
        
        %compute objective value
        obj(iter)=log_obj(y, X, B)-lambda/2*sum(B(:).^2);
        
        tr_acc(iter)=cal_te_acc(B, y, X);
        te_acc(iter)=cal_te_acc(B, te_y, te_X);
        
        %print itermediate result
        if (iter<10 || mod(iter,100)==0)
            fprintf('Iter=%d, Obj=%f, tr_acc=%f, te_acc=%f\n',   iter, obj(iter), tr_acc(iter), te_acc(iter));
        end
        
        iter=iter+1;        
    end     

end