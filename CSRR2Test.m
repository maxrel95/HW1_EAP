% csr r2 test
clear
close all 
clc

addpath( genpath( 'csr' ) )

testasset = xlsread( "testassets.xlsx" );
testasset = testasset( :, 2:end );

FF = xlsread( "FFFactors.xlsx" );
FF = FF( :, 2:end );

FFStar = xlsread( "FFFactorsStar.xlsx" );
FFStar = FFStar( :, 2:end );

% [R2, pval1, pval2a, pval2b, pval3a, pval3b, rse,gamma, trat1, trat2, trat3, trat4, ...
%           lambda, trat1a, trat2a, trat3a, trat4a, csrt, pval4,pval5] = csrgls( testasset, FF, 0);
R = testasset;
delete table1.out
diary table1.out
nlag = 0;
N = size(R,2);
fprintf(' Period:  1963:1-2021:12\n')
fprintf(' Number of lags = %2.0f\n',nlag)
fprintf(' Number of assets = %2.0f\n',N)
BigF = [FF FFStar];
modelind = NaN(2, 3); % nbr of facteur le 3
modelind(1,1:3) = [1 2 3];                   % CAPM 
modelind(2,1:3) = [4 5 6];           % C-LAB
nmodel = 2;
R2 = zeros(nmodel,1);
csrta = zeros(nmodel,1);
csrtb = zeros(nmodel,1);
pval1a = zeros(nmodel,1);
pval1b = zeros(nmodel,1);
pval2a = zeros(nmodel,1);
pval2b = zeros(nmodel,1);
pval3a = zeros(nmodel,1);
pval3b = zeros(nmodel,1);
pval4a = zeros(nmodel,1);
pval5a = zeros(nmodel,1);
pval4b = zeros(nmodel,1);
pval5b = zeros(nmodel,1);
rse = zeros(nmodel,1);
nopar = zeros(nmodel,1);
for ii=1:2
    if ii==1
       fprintf(' OLS CSR\n')
       fcn = 'csrw';
    else
       fprintf('\n GLS CSR\n')
       fcn = 'csrgls';
    end
    for i=1:nmodel
        m = modelind(i,:);
        m(isnan(m)) = [];
        nopar(i) = length(m);
        F = BigF(:,m);
        [R2(i), pval1a(i), pval1b(i), pval2a(i),pval2b(i),pval3a(i),pval3b(i),rse(i),gamma,trat1,trat2,trat3,trat4,lambda,trat1a,...
           trat2a,trat3a,trat4a,csrta(i),pval4a(i),pval5a(i),csrtb(i),pval5a(i),pval5b(i)] = feval(fcn,R,F,nlag);
    end
    fprintf('\n                                                CROSS-SECTIONAL R^2\n')
    fprintf(' ______________________________________________________________________________________________\n')
    fprintf(' Model                     FF3    FF3Star     \n')
    fprintf(' ______________________________________________________________________________________________\n')
    fprintf(' Sample R2               %6.3f   %6.3f\n',R2)
    fprintf(' p(R2=1,H1)              %6.3f   %6.3f\n',pval1b)
    fprintf(' Tests of R2=0:\n')
    fprintf(' p(R2=0,H0)              %6.3f   %6.3f\n',pval2a)
    fprintf(' se(R2)                  %6.3f   %6.3f   \n',rse)

    fprintf(' Number of Par.          %6.0f   %6.0f\n',nopar)
end

%%%%%%%%%%
delete table4.out
diary table4.out
fprintf(' Period:  1959:2-2007:7\n')
fprintf(' Number of assets = %2.0f\n',N)
fprintf(' Number of lags = %2.0f\n',nlag)
[T,N] = size(R);

modeltext = ['FF3    '; 'FF3Star'];
for ii=1:2
    if ii==1
       fprintf(' OLS\n')
    else
       fprintf('\n GLS\n')
    end
    r2diff = zeros(nmodel-1);
    pvalue = zeros(nmodel-1);    % Linear combination of chi-squared test, correctly specified 
    pvalue1 = zeros(nmodel-1);   % Misspscification robust test
    pvalue2 = zeros(nmodel-1);   % Sequential test, 2 chi-squared test then normal test
    for i=1:nmodel-1
        m1 = modelind(i,:);
        m1(isnan(m1)) = [];
        for j=i+1:nmodel
            m2 = modelind(j,:);
            m2(isnan(m2)) = [];
%
%       Check whether it is nested or nonnested
%        
            if all(ismember(m1,m2))||all(ismember(m2,m1))
               if ii==1
                  [r2d,pval2b,pval4] = nested(R,BigF,m1,m2,nlag,eye(N));      % Nested models
               else
                  [r2d,pval2b,pval4] = nested(R,BigF,m1,m2,nlag);             % Nested models
               end
               pvalue2(i,j) = pval4;
            else 
               if ii==1
                  [r2d,pval1a,pval1b,pval2a,pval2b,pval3a,pval3b,pval4] = nonnested(R,BigF,m1,m2,nlag,eye(N));   % Nonnested models
               else
                  [r2d,pval1a,pval1b,pval2a,pval2b,pval3a,pval3b,pval4] = nonnested(R,BigF,m1,m2,nlag);   % Nonnested models
               end
%              sequential test
               if pval1a>0.05
                  pvalue2(i,j) = pval1a;
               elseif pval2a>0.05
                  pvalue2(i,j) = pval2a;
               else
                  pvalue2(i,j) = pval4;
               end
            end   
            r2diff(i,j) = r2d;
            pvalue(i,j) = pval2b;
            pvalue1(i,j) = pval4;
        end
    end
    fprintf(' Difference in R2s\n')
    fprintf(' Model     FF3Star\n')
    for i=1:nmodel-1
        fprintf(' %7s',modeltext(i,:))
        for j=1:i-1
            fprintf('        ')
        end
        for j=i+1:nmodel
            fprintf('%8.3f',r2diff(i,j))
        end
        fprintf('\n')
    end
    fprintf('\n p-value under correctly specified models\n')
    fprintf(' Model     FF3Star\n')
    for i=1:nmodel-1
        fprintf(' %7s',modeltext(i,:))
        for j=1:i-1
            fprintf('        ')
        end
        for j=i+1:nmodel
            fprintf('%8.3f',pvalue(i,j))
        end
        fprintf('\n')
    end
    fprintf('\n p-value under misspecified models (using normal tests for non-nested models)\n')
    fprintf(' Model     FF3Star\n')
    for i=1:nmodel-1
        fprintf(' %7s',modeltext(i,:))
        for j=1:i-1
            fprintf('        ')
        end
        for j=i+1:nmodel
            fprintf('%8.3f',pvalue1(i,j))
        end
        fprintf('\n')
    end
    fprintf('\n p-value under misspecified models (using sequential tests for non-nested models)\n')
    fprintf(' Model    FF3Star\n')
    for i=1:nmodel-1
        fprintf(' %7s',modeltext(i,:))
        for j=1:i-1
            fprintf('        ')
        end
        for j=i+1:nmodel
            fprintf('%8.3f',pvalue2(i,j))
        end
        fprintf('\n')
    end
    reject1 = pvalue1(1:end,1:end)<0.05;
    reject2 = pvalue2(1:end,1:end)<0.05;
    fprintf('\n Number of Additional Rejections Using Normal Tests = %2.0f\n',sum(sum(reject1-reject2)))
end
diary off