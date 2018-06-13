function [P,Q,E,obj] = RSLDA(X,label,lambda1,lambda2,dim,mu,rho,Max_iter)

% code is written by Jie Wen
% If any problems, please contact: wenjie@hrbeu.edu.cn
% Please cite the reference:
% Wen J, Fang X, Cui J, et al. Robust Sparse Linear Discriminant Analysis[J]. 
% IEEE Transactions on Circuits and Systems for Video Technology, 2018,
% doi: 10.1109/TCSVT.2018.2799214

[m,n] = size(X);
max_mu = 10^5;
regu = 10^-5;
[Sw, Sb] = ScatterMat(X, label);
options = [];
options.ReducedDim = dim;
[P1,~] = PCA1(X',options);              % 这个PCA只为初始化一个正交矩阵P
%%------------------------------initilzation-------------------------------
Q = ones(m,dim);
E = zeros(m,n);
Y = zeros(m,n);
v=sqrt(sum(Q.*Q,2)+eps);
D=diag(0.5./(v));
%%-------------------------end of initilazation----------------------------
for iter = 1:Max_iter
    % P
    if (iter == 1)
        P = P1;
    else
        M = X-E+Y/mu;
        [U1,S1,V1] = svd(M*X'*Q,'econ');
        P = U1*V1';
        clear M;
    end
    % Q
    M = X-E+Y/mu;
    Q1 = 2*(Sw-regu*Sb)+lambda1*D+mu*X*X';
    Q2 = mu*X*M'*P;
    Q = Q1\Q2;
    v=sqrt(sum(Q.*Q,2)+eps);
    D=diag(1./(v));
    % E
    eps1 = lambda2/mu;
    temp_E = X-P*Q'*X+Y/mu;
    E = max(0,temp_E-eps1)+min(0,temp_E+eps1);
    % Y,mu
    Y = Y+mu*(X-P*Q'*X-E);
    mu = min(rho*mu,max_mu);
    leq = X-P*Q'*X-E;
    EE = sum(abs(E),2);
    obj(iter) = trace(Q'*(Sw-regu*Sb)*Q)+lambda1*sum(v)+lambda2*sum(EE);
    if iter >2
        if norm(leq, Inf) < 10^-7 && abs(obj(iter)-obj(iter-1))<0.00001
            iter
            break;
        end 
    end
end


