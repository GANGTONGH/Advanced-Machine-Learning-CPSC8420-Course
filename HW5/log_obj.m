function obj=log_obj(y, X, B)
% compute objective value    

    K=size(B,2)+1;
    n=length(y);
    
    XB=X*B;
    obj_1=log(sum(exp(XB), 2)+1);   %N by one vector   

    I=find(y~=K);
    J=y(I);
    idx=sub2ind([n,K-1], I, J);
    obj_2=zeros(n,1);
    obj_2(I)=XB(idx);
    
    obj=sum(obj_2-obj_1);    
    
end