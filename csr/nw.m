%
%    nw.m
%    Newey-West estimate of covariance matrix with
%    possible automatic lag selection
%    h: a Txr matrix of data
%    lag: number of lags (when lag is an empty matrix 
%    or lag<0, it is for automatic lag selection)
%    prewhite: an indicator of whether to do pre-whitening
%              or not (default is no)
%    V: covariance matrix of h
%
function V = nw(h,lag,prewhite)
if nargin<3
   prewhite = 0;
end
[T,r] = size(h);
if nargin==2&&~isempty(lag)&&lag>=0
   V = (h'*h)./T;
   for i=1:lag    
       V1 = (h(i+1:T,:)'*h(1:T-i,:))./T;
       V = V+(1-i/(lag+1))*(V1+V1');
   end
else
%
%   Automatic lag selection
%   First step: pre-whitening by fitting a VAR(1)
%
   if prewhite
      h0 = h(1:T-1,:);
      h1 = h(2:T,:);
      A = h0\h1;   % Note that our A is A' in Newey West (1994)
      he = h1-h0*A;
   else
      he = h;
   end
   T1 = length(he);
   n = fix(12*(0.01*T)^(2/9));
%
%  Compute autocorrelation coefficients of w'he
%
   w = ones(r,1);
   hw = he*w;
   sigmah = zeros(n,1);
   for i=1:n
       sigmah(i) = (hw(1:T1-i)'*hw(i+1:T1))./T1;
   end
   sigmah0 = (hw'*hw)./T1;
%
%  Compute s0 and s1 and set bandwidth parameter
%
   s0 = sigmah0+2*sum(sigmah);
   s1 = 2*([1:n]*sigmah);
   gam = 1.1447*abs(s1/s0)^(2/3);
   m = fix(gam*T^(1/3));
   V = (he'*he)./T1;
   for i=1:m
       V1 = (he(i+1:T1,:)'*he(1:T1-i,:))./T1;
       V = V+(1-i/(m+1))*(V1+V1');
   end
   if prewhite
      IA = inv(eye(r)-A'); 
      V = IA*V*IA';
   end
end
