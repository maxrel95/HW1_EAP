%
% linchi2.m	Date: 2/13/2004
% This Matlab program computes the probability that a linear combination
% of central chi-squared distribution (with 1 d.f.) is less than a constant c.
% d0: weights of the linear combination
% e: accuracy (1e-6 by default)
% delta: step size
% U: upper truncation point
%
function y = linchi2(c,d0,e)
if all(d0>=0)&&c<=0
   y = 0;
   return
end
if all(d0<=0)&&c>=0
   y = 1;
   return
end
%
%	Sorting and normalization
%
d = sort(d0);
d(d==0) = [];		% Get rid of zero elements
n = length(d);
if d(1)==d(n) 	% Use chi-squared distribution when d(1)=d(n)
   if d(1)>0
      y = chi2cdf(c/d(1),n);
   else
      y = 1-chi2cdf(c/d(1),n);
   end
   return
end
if c==0     		% Use F-distribution for this special case
   n1 = sum(d==d(1));
   n2 = sum(d==d(n));
   if n1+n2==n
      y = fcdf(-d(1)*n1/(d(n)*n2),n2,n1);
      return
   end
end
if n==2&&prod(d)>=0   % Use noncentral chi-squared
   if d(1)>0
      d1 = sum(sqrt(d))^2*c/(4*d(1)*d(2));
      d2 = d1-c/sqrt(d(1)*d(2));
      y = ncx2cdf(d1,2,d2)-ncx2cdf(d2,2,d1);
   else
      d = abs(d);
      c = abs(c);
      d1 = sum(sqrt(d))^2*c/(4*d(1)*d(2));
      d2 = d1-c/sqrt(d(1)*d(2));
      y = 1-ncx2cdf(d1,2,d2)+ncx2cdf(d2,2,d1);
   end
   return
end
dmax = max(abs(d));
d = d./dmax;
c = c/dmax;
absd = abs(d);
if size(d,1)==1
   d = d';
end
if nargin<=2
   e = 1e-6;
end
ei = log(0.1*e);	% log of integration error
et = log(0.9*e);	% log of truncation error
%
%	Choosing delta and K for integration
%
psi1 = 0;
psi2 = 0;
mar = 1e-9;
OPTIONS = optimset('TolFun',log(1.1));
if d(1)<0
   t1 = fzero(@lin1,[1/(2*d(1))+mar -mar],OPTIONS);
   psi1 = sum(d./(1-2*t1*d));
% When c is a very large negative number, no need to worry P[X<c-2*pi/delta]<E_I 
%	but need to worry P[X<c+2*pi/delta]<E_I.
end
if d(end)>0
   t2 = fzero(@lin1,[mar 1/(2*d(end))-mar],OPTIONS);
   psi2 = sum(d./(1-2*t2*d));
end
delta = 2*pi*min(1/max(c-psi1,eps),1/max(psi2-c,eps));
%
%	Imhof truncation bound
%
sd = sum(log(absd));
a = exp(-(sd+2*et)/n);	% To avoid overflow problem
U1 = 0.5*a*((2/(n*pi))^(2/n));
%
%	AKS truncation bound
%
OPTIONS1 = optimset('fzero');
d2 = d.*d;
U2 = abs(fzero(@(u) lin2(u,exp(et)),1,OPTIONS1));
%
%	Lu and King truncation bound
%
n1 = n-0.5;
a = exp(-(sd+0.25*log(sum(1./d2))+2*et)/n1);		% To avoid overflow problem
U3 = 0.5*a*((2^(0.75)/(n1*pi))^(2/n1));
UU = min([U1 U2 U3]);
K = ceil(UU/delta-0.5);
%
%	Consider further splitting in case of large K
%
if K>5000
   [Umin,ii] = min(UU);
   V = zeros(1,9);
   alpha = [1:9]/10;
   et2 = log(alpha)+et;
   if ii==1
      a = exp(-(sd+2*et2)/n);	% To avoid overflow problem
      U = 0.5*a*((2/(n*pi))^(2/n));
   end
   if ii==2
      U = zeros(1,9);   
      for i=1:9
          U(i) = abs(fzero(@(u) lin2(u,exp(et2(i))),1,OPTIONS1));
      end
   end
   if ii==3
      a = exp(-(sd+0.25*log(sum(1./d2))+2*et2)/n1);	% To avoid overflow problem
      U = 0.5*a*((2^(0.75)/(n1*pi))^(2/n1));
   end
   for i=1:9
       V(i) = fzero(@lin3,[1e-12 1-1e-12]*U(i),OPTIONS1);
   end
   UU = min([Umin V]);
   K = ceil(UU/delta-0.5);
end
if K>3000000
   fprintf(' K = %10.0f, should consider reducing the precision.\n',K)
   y = NaN;
else
   index = 0.5:1:K+0.5;
   u = index*delta;
   du = 2*d*u;
   thetak = 0.5*sum(atan(du))-c*u;
   gk = prod(1+du.*du).^(1/4);
   y = 0.5-sum(sin(thetak)./(gk.*index))/pi;
   y = min(max(y,0),1);
end

function y = lin1(t)
   x = 1-2*t*d;
   psi3 = 0.5*sum(log(x));
   psi4 = sum(d./x);
   y = psi3+t*psi4+ei;
end

function y = lin2(u,P1)
   x = 1+4*u*u*d2;
   y = sum(log(x))-log(1+(2/(pi*P1))^4);
end

function y = lin3(v)
   x = 1+4*v*v*d2;
   y = 0.25*sum(log(x))+log(1-alpha(i))+et-log(log(U(i)/v));
end

end
