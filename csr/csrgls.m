%
%   	CSRGLS.M
% This Matlab performs the two-pass GLS CSR.
%
% Input:
% R: Returns on N test assets
% F: K systematic factors
% lag: Number of lags of Newey-West adjustment (default is 0)
%      For automatic lag selection, set lag to be an empty matrix.
%
% Output:
% R2: Sample CSR R^2
% pval1a: p-value of testing R^2=1 (imposing H0: R^2=1)
% pval1b: p-value of testing R^2=1 (without imposing H0: R^2=1)
% pval2a: p-value of testing R^2=0 (imposing H0: gamma_1=0_K)
% pval2b: p-value of testing R^2=0 (without imposing H0: gamma_1=0_K) 
% pval3a: p-value of Wald test of H0: gamma_1=0_K (imposing H0: gamma_1=0_K)
% pval3b: p-value of Wald test of H0: gamma_1=0_K (without imposing H0: gamma_1=0_K)
% rse: Standard error of sample CSR R^2 (when 0 < R^2 < 1)  
% gamma: zero-beta rate and risk premia
% trat1: Fama-MacBeth t-ratio
% trat2: Shanken EIV t-ratio
% trat3: EIV t-ratio under general distribution assumptions
% trat4: misspecification robust t-ratio
% lambda: zero-beta rate and price of covariance risk
% trat1a: Fama-MacBeth t-ratio
% trat2a: Shanken EIV t-ratio
% trat3a: EIV t-ratio under general distribution assumptions
% trat4a: misspecification robust t-ratio
% csrta: Shanken's CSRT statistic (imposing H0: R^2=1)
% pval4a: asymptotic p-value of CSRT statistic (imposing H0: R^2=1)
% pval5a: approximate finite sample p-value of CSRT statistic (imposing H0: R^2=1)
% csrtb: Shanken's CSRT statistic (without imposing H0: R^2=1)
% pval4b: asymptotic p-value of CSRT statistic (without imposing H0: R^2=1)
% pval5b: approximate finite sample p-value of CSRT statistic (without imposing H0: R^2=1)
%
function [R2,pval1a,pval1b,pval2a,pval2b,pval3a,pval3b,rse,gamma,trat1,trat2,trat3,trat4, ...
          lambda,trat1a,trat2a,trat3a,trat4a,csrta,pval4a,pval5a,csrtb,pval4b,pval5b] = csrgls(R,F,lag)
[T,N] = size(R);
if nargin<3
   lag = 0;
end
index = any(isnan(F),2);
F(index,:) = [];
[T,K] = size(F);
R(index,:) = [];
Y = [F R];
mu1 = mean(F)';
mu2 = mean(R)';
V = cov(Y,1);
V11 = V(1:K,1:K);
V12 = V(1:K,K+1:end);
V21 = V12';
V22 = V(K+1:end,K+1:end);
Sigma = V22-V21*inv(V11)*V12;
b = V21*inv(V11);
X = [ones(N,1) b];
W = inv(V22);
H = inv(X'*W*X);
A = H*X'*W;
gamma = A*mu2;
gamma1 = gamma(2:end);
e = mu2-X*gamma;
g0 = sum(W*mu2)/sum(sum(W));
e0 = mu2-g0;
Q0 = e0'*W*e0;
Q1 = e'*W*e;
R2 = 1-Q1/Q0;
Fd = F-ones(T,1)*mu1';     % De-meaned factors
Rd = R-ones(T,1)*mu2';     % De-meaned returns
wt = Fd*inv(V11)*gamma1;
yt = 1-wt;                 % SDF
%
%   Specification test
%
Wh = sqrtm(W);
WP = Wh*null(X'*Wh);
g0 = (R*WP).*(yt*ones(1,N-K-1));
S0 = nw(g0,lag);
xi = -eig(S0)./Q0;       % S is actually P'W^\frac{1}{2}SW^\frac{1}{2}P
pval1a = linchi2(T*(R2-1),xi);    % p-val for specification test, imposing H0: R^2=1
E = Rd-Fd*b';
g = (E*WP).*(yt*ones(1,N-K-1));
S = nw(g,lag);
xi = -eig(S)./Q0;       % S is actually P'W^\frac{1}{2}SW^\frac{1}{2}P
pval1b = linchi2(T*(R2-1),xi);    % p-val for specification test, without imposing H0: R^2=1
%
%   Test of R^2=0
%
gammat = Rd*A';    % This is \hat\gamma_t-\hat\gamma
phit = gammat-[zeros(T,1) Fd];  % This is \hat\phi
ut = ((T-N-2)/T)*Rd*W*e;
zt = [zeros(T,1) Fd*inv(V11)];
ht0 = gammat+(zt*H-gammat).*(ut*ones(1,K+1));   % Imposing H0: gamma_1=0_K
ht1 = gammat-phit.*(wt*ones(1,K+1));     % Without misspecification adjustment
ht2 = ht0-phit.*(wt*ones(1,K+1));        % With misspecification adjustment
V0 = nw(ht0,lag);
V1 = nw(ht1,lag);
V2 = nw(ht2,lag);
H22i = inv(H(2:K+1,2:K+1));
H22iq = sqrtm(H22i);
mat = H22iq*V0(2:K+1,2:K+1)*H22iq;
xi = eig((0.5/Q0)*(mat+mat'));   
pval2a = 1-linchi2(T*R2,xi);
mat = H22iq*V2(2:K+1,2:K+1)*H22iq;
xi = eig((0.5/Q0)*(mat+mat'));   
pval2b = 1-linchi2(T*R2,xi);
%
%   Wald test of H0: gamma_1=0_K
%
wald0 = T*gamma1'*inv(V0(2:K+1,2:K+1))*gamma1;
pval3a = 1-chi2cdf(wald0,K);
wald = T*gamma1'*inv(V1(2:K+1,2:K+1))*gamma1;
pval3b = 1-chi2cdf(wald,K);
%
%   Standard error of sample R^2.
%
vt = ((T-N-2)/T)*Rd*W*e0;
nt = (ut.*ut-2*ut.*yt+(1-R2)*(2*vt-vt.*vt))./Q0;
vn = nw(nt,lag);
rse = sqrt(vn/T);
%
%   t-ratios
%
VFM = A*V22*A';         % Fama-MacBeth
VS0 = (1+gamma1'*inv(V11)*gamma1)*(A*Sigma*A');
VS = VS0+[zeros(1,K+1); zeros(K,1) V11];   % Shanken EIV, zero factor autorrelations
Vf = nw(Fd,lag);
trat1 = gamma./sqrt(diag(VFM)./T);
trat2 = gamma./sqrt(diag(VS)./T);
trat3 = gamma./sqrt(diag(V1)./T);
trat4 = gamma./sqrt(diag(V2)./T);
%
%   Computation of lambda
%
lambda1 = inv(V11)*gamma1;
lambda = [gamma(1); lambda1];
C = [ones(N,1) V21];
H1 = inv(C'*W*C);
A1 = H1*C'*W;
VFM = A1*V22*A1';       % Fama-MacBeth
cc = 1+lambda1'*V11*lambda1;
VS0 = cc*A1*Sigma*A1';
VS = VS0+[zeros(1,K+1); zeros(K,1) cc*inv(V11)+lambda1*lambda1'];    % Assume i.i.d. normality
h2 = (Fd*inv(V11)).*(yt*ones(1,K))+ones(T,1)*lambda1';
Vh2 = nw(h2,lag);
lambdat = Rd*A1';   % This is \hat\lambda_t-\hat\lambda
ht1 = lambdat.*(yt*ones(1,K+1))+ones(T,1)*[0 lambda1'];   % Without misspecification adjustment
ht2 = ht1+([zeros(T,1) Fd]*H1-lambdat).*(ut*ones(1,K+1)); % With misspecification adjustment
V1 = nw(ht1,lag);
V2 = nw(ht2,lag);
trat1a = lambda./sqrt(diag(VFM)./T);
trat2a = lambda./sqrt(diag(VS)./T);
trat3a = lambda./sqrt(diag(V1)./T);
trat4a = lambda./sqrt(diag(V2)./T);
%
%   Computation of generalized Shanken (1985) test statistics 
%
e1 = WP'*e;
csrta = e1'*inv(S0)*e1 ;
pval4a = 1-chi2cdf(T*csrta,N-K-1);
pval5a = 1-fcdf(csrta*(T-N+1)/(N-K-1),N-K-1,T-N+1);
csrtb = e1'*inv(S)*e1 ;
pval4b = 1-chi2cdf(T*csrtb,N-K-1);
pval5b = 1-fcdf(csrtb*(T-N+1)/(N-K-1),N-K-1,T-N+1);
