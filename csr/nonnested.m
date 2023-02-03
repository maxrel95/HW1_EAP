%
%  nonnested.m
%  This MATLAB M-file computes the difference of cross-sectional R^2
%  of two nonnested models, as welll as its p-value under different
%  assumptions on the null hypothesis of H0: \rho_1^=\rho_2^2
%  the models are correctly specified and misspecified.
%  Input:
%  R: returns on N test assets 
%  BigF: set of all factors
%  m1: index for model 1
%  m2: index for model 2 
%  lag: Number of lag adjustments for computing Newey-West standard error 
%      (default is lag=0)
%  W: weighting matrix (default is GLS, W=\hat{V}_{22}^{-1})
%
%  Output:
%  rsqd: difference of sample R^2 of models 1 and 2
%  pval1a: p-value of testing H0: y1=y2 using chi-squared test
%  pval1b: p-value of testing H0: y1=y2 using linear combination of chi-squared test
%  pval2a: p-value of testing H0: rho_1^2=rho_2^2=1 using chi-squared test (imposing rho^2=1)
%  pval2b: p-value of testing H0: rho_1^2=rho_2^2=1 using linear combination of chi-squared test (imposing rho^2=1)
%  pval3a: p-value of testing H0: rho_1^2=rho_2^2=1 using chi-squared test (not imposing rho^2=1)
%  pval3b: p-value of testing H0: rho_1^2=rho_2^2=1 using linear combination of chi-squared test (not imposing rho^2=1)
%  pval4: p-value of testing H0: 0<rho_1^2=rho_2^2<1 using normal test
%
function [rsqd,pval1a,pval1b,pval2a,pval2b,pval3a,pval3b,pval4] = nonnested(R,BigF,m1,m2,lag,W)
if nargin<5
   lag = 0;
end
F1 = BigF(:,intersect(m1,m2));
[c,ia,ib] = setxor(m1,m2);
F2 = BigF(:,m1(ia));
F3 = BigF(:,m2(ib));
K1 = size(F1,2);
K2 = size(F2,2);
K3 = size(F3,2);
K = K1+K2+K3 ;
Y = [F1 F2 F3 R] ;
index = any(isnan(Y),2);
F1(index,:) = [];
F2(index,:) = [];
F3(index,:) = [];
R(index,:) = [];
N = size(R,2);
Y(index,:) = [];
T = length(Y);
mu2 = mean(R)';
Rd = R-ones(T,1)*mu2';          % de-meaned returns
F1d = [F1-ones(T,1)*mean(F1) F2-ones(T,1)*mean(F2)];    % de-meaned F1 and F2
F2d = [F1d(:,1:K1) F3-ones(T,1)*mean(F3)];    % de-meaned F1 and F3
GLS = 0;
if nargin<6
   GLS = 1;
   W = inv(cov(R,1));
end
V21a = (R'*F1d)./T;
V21b = (R'*F2d)./T;
C1 = [ones(N,1) V21a];
C2 = [ones(N,1) V21b];
Q0 = mu2'*W*mu2-sum(W*mu2)^2/sum(sum(W));
Q1 = mu2'*W*mu2-mu2'*W*C1*inv(C1'*W*C1)*C1'*W*mu2;
Q2 = mu2'*W*mu2-mu2'*W*C2*inv(C2'*W*C2)*C2'*W*mu2;
rsqd = (Q2-Q1)/Q0;
H1 = inv(C1'*W*C1);
H1i = inv(H1(K1+2:end,K1+2:end));
A1 = H1*C1'*W;
H2 = inv(C2'*W*C2);
H2i = inv(H2(K1+2:end,K1+2:end));
A2 = H2*C2'*W;
lambda = A1*mu2;
eta = A2*mu2;
e1 = mu2-C1*lambda;
e2 = mu2-C2*eta;
e0 = (eye(N)-ones(N,1)*inv(ones(1,N)*W*ones(N,1))*ones(1,N)*W)*mu2 ;
y1t = 1-F1d*lambda(2:end);
y2t = 1-F2d*eta(2:end);
lambda2 = lambda(K1+2:end);
eta2 = eta(K1+2:end);
u1t = Rd*W*e1;
u2t = Rd*W*e2;
vt = Rd*W*e0;
%
%   Tests of y1=y2
%
psi = [lambda2; eta2];
lambdat = Rd*A1(K1+2:end,:)'; % lambdat is \hat\lambda_{2t}-\hat\lambda_2
etat = Rd*A2(K1+2:end,:)';    % etat is \hat\eta_{2t}-\hat\eta_2
ht1 = [lambdat.*(y1t*ones(1,K2))+ones(T,1)*lambda2' etat.*(y2t*ones(1,K3))+ones(T,1)*eta2'];
if GLS
   u1t = ((T-N-2)/T)*u1t;
   u2t = ((T-N-2)/T)*u2t; 
   ht2 = ht1+[([zeros(T,1) F1d]*H1(:,K1+2:end)-lambdat).*(u1t*ones(1,K2)) ...
         ([zeros(T,1) F2d]*H2(:,K1+2:end)-etat).*(u2t*ones(1,K3))];    % With misspecification adjustment
else
   ht2 = ht1+[([zeros(T,1) F1d]*H1(:,K1+2:end)).*(u1t*ones(1,K2)) ...
         ([zeros(T,1) F2d]*H2(:,K1+2:end)).*(u2t*ones(1,K3))];         % With misspecification adjustment
end
V1 = nw(ht2,lag);
wald = T*psi'*inv(V1)*psi;
pval1a = 1-chi2cdf(wald,K2+K3);
xi = eig([H1i zeros(K2,K3); zeros(K3,K2) -H2i]*V1)./Q0;
pp = linchi2(T*rsqd,xi);
pval1b = 2*min(pp,1-pp);
%
%   Tests of rho1^2=rho_2^2=1
%
Whalf = sqrtm(W);
P1 = null(C1'*Whalf);
P2 = null(C2'*Whalf);
n1 = N-K1-K2-1;
n2 = N-K1-K3-1;
g1 = R.*(y1t*ones(1,N));
g2 = R.*(y2t*ones(1,N));
g0 = [g1*Whalf*P1 g2*Whalf*P2];
S0 = nw(g0,lag);
ee = [P1'*Whalf*e1; P2'*Whalf*e2];
chi2 = T*ee'*inv(S0)*ee;
pval2a = 1-chi2cdf(chi2,n1+n2);
S0(1:n1,1:end) = -S0(1:n1,1:end);
xi = eig(S0)./Q0;    
pp = linchi2(T*rsqd,xi);
pval2b = 2*min(pp,1-pp);
E1 = Rd-F1d*inv((F1d'*F1d)./T)*V21a';    % Regression residuals
E2 = Rd-F2d*inv((F2d'*F2d)./T)*V21b';
g1 = E1.*(y1t*ones(1,N));
g2 = E2.*(y2t*ones(1,N));
g = [g1*Whalf*P1 g2*Whalf*P2];
S = (g'*g)./T;
chi2 = T*ee'*inv(S)*ee;
pval3a = 1-chi2cdf(chi2,n1+n2);
S(1:n1,1:end) = -S(1:n1,1:end);
xi = eig(S)./Q0;    
pp = linchi2(T*rsqd,xi);
pval3b = 2*min(pp,1-pp);
%
%    Test of 0<rho_1^2=rho_2^2<1
%
dt = 2*(-u1t.*y1t+(Q1./Q0)*vt)-2*(-u2t.*y2t+(Q2./Q0)*vt) ;
if GLS
   vt = ((T-N-2)/T)*vt; 
   dt = (u1t.^2-2*y1t.*u1t+(Q1./Q0)*(2*vt-vt.^2))-(u2t.^2-2*y2t.*u2t+(Q2./Q0)*(2*vt-vt.^2)) ;
end
dt = dt./Q0;
vd = nw(dt,lag);
pval4 = 2*(1-normcdf(abs(rsqd)/sqrt(vd./T)));
