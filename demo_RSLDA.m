% code is written by Jie Wen
% If any problems, please contact: wenjie@hrbeu.edu.cn
% Please cite the reference:
% Wen J, Fang X, Cui J, et al. Robust Sparse Linear Discriminant Analysis[J]. 
% IEEE Transactions on Circuits and Systems for Video Technology, 2018,
% doi: 10.1109/TCSVT.2018.2799214
clc
clear all;
clear memory;

name = 'YaleB_32x32_98'
load (name);
fea = double(fea);
sele_num = 15;                                            %  select training samples 

nnClass   = length(unique(gnd));                          % The number of classes;
num_Class = [];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];           % The number of samples of each class
end

Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass
    idx = find(gnd==j);   
    randIdx = randperm(num_Class(j));  % radomly select the sele_num training samples
    Train_Ma  = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];   % Random select select_num samples per class for training
    Train_Lab = [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma   = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  %Random select remaining  samples per class for test
    Test_Lab  = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]);

% -------------   set parameters ---------------- %
lambda1 = 0.00001;
lambda2 = 0.0001;
dim = 115;
mu  = 0.1;
rho = 1.01;
max_iter = 100;

[P,Q,E,obj] = RSLDA(Train_Ma,Train_Lab,lambda1,lambda2,dim,mu,rho,max_iter);

Test_Maa  = Q'*Test_Ma;
Train_Maa = Q'*Train_Ma;
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
[class_test] = knnclassify(Test_Maa',Train_Maa',Train_Lab,1,'euclidean','nearest');
acc_test = sum(Test_Lab == class_test)/length(Test_Lab)*100