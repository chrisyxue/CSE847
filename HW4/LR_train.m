function [weights] = LR_train(data, labels, epsilon, maxiter)
n_sample=size(data,1);
n_feature=size(data,2);
weights=zeros(n_feature,1);

n_iter=1;
new_cost=1;
while n_iter<maxiter
    old_cost=new_cost;
    z=data*weights;
    label_estimated=1./(1+exp(-z));
    new_cost = (-1/n_sample)*(sum((labels.*log(label_estimated)) + ((1-labels).*(log(1-label_estimated)))));
    d_cost=abs(new_cost-old_cost);
    if(d_cost<epsilon)
        break;
    end
    n_iter=n_iter+1;
    d_weights = (1/n_sample)*(data'*(label_estimated-labels));
    weights=weights-d_weights;
end