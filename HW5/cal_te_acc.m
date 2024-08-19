function acc=cal_te_acc(B, te_label, te_data)

% calculate the accuracy

    n_te=size(te_data,1);
    w=te_data*B;
    w=[w, zeros(n_te,1)];
    [w_v, pred_label]=max(w,[],2);
    acc=1-sum(pred_label~=te_label)/n_te;
 
    
end