% MAIN
% Version 28-March-2020
clear;
clc;
close all;

fig=figure;
set(fig,'DefaultAxesFontSize',18);
set(fig,'DefaultAxesFontWeight','bold');
set(fig,'PaperSize',[17 10]);

data = load('data.txt');
labels = load('labels.txt');

data = cat(2,data,ones(4601,1)); % Add Bias
  
train_num =[200,500,800,1000,1500,2000];

test_data=data(2001:4601,:);
test_label=labels(2001:4601);
  
epsilon=1e-5;
maxiter=1000;

test_acc_list=zeros(1,length(train_num));
for ii=1:length(train_num)
    train_data = data(1:train_num(ii),:);
    train_label = labels(1:train_num(ii));
    weight = LR_train(train_data,train_label,epsilon,maxiter);
    test_label_estimated = test_data*weight;
    test_label_estimated = 1./(1+exp(-test_label_estimated));
    
    % Sign Function
    test_label_estimated(test_label_estimated>=0.5) = 1;
    test_label_estimated(test_label_estimated<0.5) = 0;
    
    % Store Results
    test_acc_list(ii) = mean(test_label_estimated==test_label);
end


plot(train_num, test_acc_list,'x-');
xlabel('Number of Training Samples');
ylabel('Test Acc');
box on;
title('')
set(gcf,'WindowStyle','normal','Position', [200,200,640,360]);
saveas(gcf,"fig1.pdf")


% LASSO
data = load('ad_data.mat');
par_list  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
num_features_list = zeros(size(par_list,2),1);
auc_list = zeros(size(par_list,2),1);
for ii = 1:size(par_list,2)
    par = par_list(ii);
    [w, c] = lasso_train(data.X_train, data.y_train, par);
    num_features_list(ii) = nnz(w);
    preds = data.X_test * w + c;
    [X,Y,T,auc_list(ii)] = perfcurve(data.y_test,preds,1);
end

plot(par_list, num_features_list)
xlabel('Lambda')
ylabel('Number of features')
title('')
set(gcf,'WindowStyle','normal','Position', [200,200,640,360]);
saveas(gcf,"Lambda-Features.pdf")

plot(par_list, auc_list)
xlabel('Lambda')
ylabel('AUC')
title('')
set(gcf,'WindowStyle','normal','Position', [200,200,640,360]);
saveas(gcf,"Lambda-AUC.pdf")



