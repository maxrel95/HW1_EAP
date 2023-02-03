%
%   nested.m
%   This MATLAB M-file computes the differenece of cross-sectional
%   R^2 of two nested models (with W as the weighting matrix), as well 
%   as its p-values under the assumption that the models are correctly 
%   specified and misspecified.
%
% Input:
% R: returns on N assets
% BigF: set of all factors
% m1: index for model 1
% m2: index for model 2
% lag: Number of lag adjustments for computing Newey-West standard error 
%      (default is lag=0)
% W: weighting matrix (default is GLS, W=\hat{V}_{22}^{-1})
%
% Output:
% rsqd: difference of sample R^2 of models 1 and 2
% pval1: p-value of testing H_0: rho_1^2=rho_2^2 under correctly specified model
% pval2: p-value of testing H_0: rho_1^2=rho_2^2 under misspecified model
% pval3: p-value of Wald test of H_0: rho_1^2=rho_2^2 which assumes 
%        correctly specified model.
% pval4: p-value of Wald test of H_0: rho_1^2=rho_2^2 which assumes 
%        potentially misspecified model.
%
function [rsqd,pval1,pval2,pval3,pval4] = nested(R,BigF,m1,m2,lag,W)
[T,N] = size(R);
if nargin<5
   lag = 0;
end
if length(m1)<length(m2)
   F1 = BigF(:,m1);
   F2 = BigF(:,setdiff(m2,m1));
else
   F1 = BigF(:,m2);
   F2 = BigF(:,setdiff(m1,m2));
end
index = any(isnan([F1 F2]),2);
F1(index,:) = [];
F2(index,:) = [];
R(index,:) = [];
[T,K1] = size(F1);
[T,K2] = size(F2);
K = K1+K2;
F = [F1 F2];
Y = [F R];
mu = mean(Y)';
mu1 = mu(1:K);
mu2 = mu(K+1:end);
V = cov(Y,1);
V21 = V(K+1:end,1:K);
GLS = 0;
if nargin<6
   GLS = 1;
   W = inv(V(K+1:end,K+1:end));
end
C1 = [ones(N,1) V21];
C2 = C1(:,1:K1+1);
Q0 = mu2'*W*mu2-sum(W*mu2)^2/sum(sum(W));
Q1 = mu2'*W*mu2-mu2'*W*C1*inv(C1'*W*C1)*C1'*W*mu2;
Q2 = mu2'*W*mu2-mu2'*W*C2*inv(C2'*W*C2)*C2'*W*mu2;
rsqd = (Q2-Q1)/Q0;
H1 = inv(C1'*W*C1);
H1i = inv(H1(K1+2:end,K1+2:end));
A1 = H1*C1'*W;
lambda = A1*mu2;
e = mu2-C1*lambda;
lambda2 = lambda(K1+2:end);
F1d = F-ones(T,1)*mu1';        % de-meaned factors
Rd = R-ones(T,1)*mu2';         % de-meaned returns
yt = 1-F1d*lambda(2:end);      % SDF
%lambdat = Rd*A1(K1+2:end,:)';  % lambdat is \hat\lambda_{2t}-\hat\lambda_2
lambdat = R*A1(K1+2:end,:)';  % lambdat is \hat\lambda_{2t}-\hat\lambda_2
ut = Rd*W*e;
%ht1 = lambdat.*(yt*ones(1,K2))+ones(T,1)*lambda2';        % Without misspecification adjustment
ht1 = lambdat.*(yt*ones(1,K2));                            % Without misspecification adjustment
if GLS
   %ht2 = ht1+([zeros(T,1) F1d]*H1(:,K1+2:end)-lambdat).*(ut*ones(1,K2));  % With misspecification adjustment
   ht2 = ht1+([zeros(T,1) F1d]*H1(:,K1+2:end)-lambdat).*(ut*ones(1,K2));   % With misspecification adjustment
else
   ht2 = ht1+([zeros(T,1) F1d]*H1(:,K1+2:end)).*(ut*ones(1,K2));  % With misspecification adjustment
end
V1 = nw(ht1,lag);
V2 = nw(ht2,lag);
xi = eig(H1i*V1)./Q0;
pval1 = 1-linchi2(T*rsqd,xi);
xi = eig(H1i*V2)./Q0;
pval2 = 1-linchi2(T*rsqd,xi);
%
%   Wald test
%
wald0 = T*lambda2'*inv(V1)*lambda2;
pval3 = 1-chi2cdf(wald0,K2);
wald1 = T*lambda2'*inv(V2)*lambda2;
pval4 = 1-chi2cdf(wald1,K2);
if length(m2)>length(m1)
   rsqd = -rsqd;
end
