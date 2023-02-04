##########################################
# Empirical Asset Pricing                #
# Homework 1                             #
# Maxime Borel                           #
# Date: Febuary 2023                     #              
##########################################

import pandas as pd
import numpy as np
import datetime as dt
import wrds
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
import statsmodels.api as sm

import numpy as np

def nw(h, lag=None, prewhite=False):
    T, r = h.shape
    if lag is not None and lag >= 0:
        V = (h.T @ h) / T
        for i in range(lag):
            V1 = (h[i+1:, :].T @ h[:-i, :]) / T
            V = V + (1 - i / (lag + 1)) * (V1 + V1.T)
    else:
        if prewhite:
            h0 = h[:-1, :]
            h1 = h[1:, :]
            A = np.linalg.lstsq(h0, h1, rcond=None)[0]
            he = h1 - h0 @ A
        else:
            he = h
        T1 = len(he)
        n = int(12 * (0.01 * T) ** (2 / 9))
        w = np.ones((r, 1))
        hw = he @ w
        sigmah = np.zeros((n, 1))
        for i in range(n):
            sigmah[i] = (hw[:-i].T @ hw[i:]) / T1
        sigmah0 = (hw.T @ hw) / T1
        s0 = sigmah0 + 2 * np.sum(sigmah)
        s1 = 2 * np.sum(np.arange(1, n + 1) * sigmah)
        gam = 1.1447 * abs(s1 / s0) ** (2 / 3)
        m = int(gam * T ** (1 / 3))
        V = (he.T @ he) / T1
        for i in range(m):
            V1 = (he[i+1:, :].T @ he[:-i, :]) / T1
            V = V + (1 - i / (m + 1)) * (V1 + V1.T)
        if prewhite:
            IA = np.linalg.inv(np.eye(r) - A.T)
            V = IA @ V @ IA.T
    return V


###################
# Connect to WRDS #
###################
conn=wrds.Connection( wrds_username = 'maxrel95' )

###################
# Compustat Block #
###################
# this request ask for global company key, total asset, prefered stock liquidating value
# prefered stock redeemable stock holer equity total prefered stock capital differed tax and inv tax
# from compustat fundamental annual industry not financial, standardized format domestic population src
# consolidated acccounting value 
comp = conn.raw_sql("""
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1959'
                    """, date_cols=['datadate'])

comp['year']=comp['datadate'].dt.year # creat a col with year only 

# create preferrerd stock
comp['ps']=np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps']=np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
comp['ps']=np.where(comp['ps'].isnull(),0,comp['ps'])
comp['txditc']=comp['txditc'].fillna(0)

# create book equity
comp['be']=comp['seq']+comp['txditc']-comp['ps']
comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

# number of years in Compustat
comp=comp.sort_values(by=['gvkey','datadate'])
comp['count']=comp.groupby(['gvkey']).cumcount()

comp=comp[['gvkey','datadate','year','be','count']]

###################
# CRSP Block      #
###################
# sql similar to crspmerge macro
# acces to db, crsp monthly security and msename
# id numb, shrare code, exchange code, holding period ret, holding ret without div, share outstanding
# price average bid ask
#msf = monthly stock file,msenames monthly stock event 
crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1959' and '12/31/2021'
                      and b.exchcd between 1 and 3
                      """, date_cols=['date']) 

# change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Line up date to be end of month
crsp_m['jdate']=crsp_m['date']+MonthEnd(0)

# add delisting return
#table monthly stock event delisting
dlret = conn.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """, date_cols=['dlstdt'])

dlret.permno=dlret.permno.astype(int)
dlret['jdate']=dlret['dlstdt']+MonthEnd(0)

crsp = pd.merge(crsp_m, dlret, how='left',on=['permno','jdate'])
crsp['dlret']=crsp['dlret'].fillna(0)
crsp['ret']=crsp['ret'].fillna(0)

# retadj factors in the delisting returns
crsp['retadj']=(1+crsp['ret'])*(1+crsp['dlret'])-1

# calculate market equity
crsp['me']=crsp['prc'].abs()*crsp['shrout'] 
crsp=crsp.drop(['dlret','dlstdt','prc','shrout'], axis=1)
crsp=crsp.sort_values(by=['jdate','permco','me'])

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()

# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate','permco'])['me'].max().reset_index()

# join by jdate/maxme to find the permno
crsp1=pd.merge(crsp, crsp_maxme, how='inner', on=['jdate','permco','me'])

# drop me column and replace with the sum me
crsp1=crsp1.drop(['me'], axis=1)

# join with sum of me to get the correct market cap info
crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','permco'])

# sort by permno and date and also drop duplicates
crsp2=crsp2.sort_values(by=['permno','jdate']).drop_duplicates()

# keep December market cap
crsp2['year']=crsp2['jdate'].dt.year
crsp2['quarter'] = crsp2['jdate'].dt.quarter
crsp2['month']=crsp2['jdate'].dt.month

decme=crsp2[crsp2['month'].isin([ 3, 6, 9, 12 ])] # take all decembre market cap
decme=decme[['permno','date','jdate','me','year', 'month', 'quarter']].rename(columns={'me':'dec_me'}) # lagged by 6 month the market cap
decme['ffdate'] = decme['jdate']+MonthEnd( 6 )
decme['ffyear'] = decme['ffdate'].dt.year
decme['ffmonth'] = decme['ffdate'].dt.month
decme['ffquarter'] = decme['ffdate'].dt.quarter

decme = decme[['permno','ffyear', 'ffmonth', 'ffquarter', 'dec_me']]

### July to June dates
crsp2['ffdate']=crsp2['jdate']+MonthEnd(-6) # lag the date by 6 month 
crsp2['ffyear']=crsp2['ffdate'].dt.year # get the year 
crsp2['ffmonth']=crsp2['ffdate'].dt.month # get the month 
crsp2['ffquarter']=crsp2['ffdate'].dt.quarter # get the month 
crsp2[ 'ffmonthadapted' ] = crsp2.groupby(['permno', 'ffyear', 'ffquarter']).cumcount()+1
crsp2['1+retx']=1+crsp2['retx']
crsp2=crsp2.sort_values(by=['permno','date'])

# cumret by stock
crsp2['cumretx']=crsp2.groupby(['permno','ffyear', 'ffquarter'])['1+retx'].cumprod()

# lag cumret
crsp2['lcumretx']=crsp2.groupby(['permno'])['cumretx'].shift(1)

# lag market cap
crsp2['lme']=crsp2.groupby(['permno'])['me'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
crsp2['count']=crsp2.groupby(['permno']).cumcount()
crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])

# baseline me
mebase=crsp2[ crsp2['ffmonthadapted']==1 ][ [ 'permno','ffyear', 'ffquarter', 'lme' ] ].rename(columns={'lme':'mebase'})

# merge result back together
crsp3=pd.merge(crsp2, mebase, how='left', on=['permno','ffyear', 'ffquarter'])
crsp3['wt']=np.where(crsp3['ffmonthadapted']==1, crsp3['lme'], crsp3['mebase']*crsp3['lcumretx'])

# Info as of June
crsp3_jun = crsp3[ crsp3['ffmonthadapted']==3 ]

crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno','ffyear', 'ffquarter'])
crsp_jun=crsp_jun[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','dec_me']]
crsp_jun=crsp_jun.sort_values(by=['permno','jdate']).drop_duplicates()

#######################
# CCM Block           #
#######################
ccm=conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """, date_cols=['linkdt', 'linkenddt'])

# if linkenddt is missing then set to today date
ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

ccm1=pd.merge(comp[['gvkey','datadate','be', 'count']],ccm,how='left',on=['gvkey'])
ccm1['yearend']=ccm1['datadate']+YearEnd(0)
ccm1['jdate']=ccm1['yearend']+MonthEnd(6) 

# set link date bounds
ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
ccm2=ccm2[['gvkey','permno','datadate','yearend', 'jdate','be', 'count']]

# link comp and crsp
ccm_jun=pd.merge_ordered(crsp_jun, ccm2, how='left', on=['permno', 'jdate'], fill_method='ffill')
ccm_jun['beme']=ccm_jun['be']*1000/ccm_jun['dec_me']

# positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
nyse=ccm_jun[ (ccm_jun['beme']>0) & (ccm_jun['me']>0) & \
             (ccm_jun['count']>=1) & ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))]

# size breakdown
#nyse_sz=nyse.groupby(['jdate'])['me'].quantile( q=[0.25, 0.75]).reset_index()
nyse_sz=nyse.groupby(['jdate'])['me'].describe(percentiles=[0.25, 0.75]).reset_index()
nyse_sz=nyse_sz[['jdate','25%','75%']].rename(columns={'25%':'sz25', '75%':'sz75'})

# beme breakdown
nyse_bm=nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.25, 0.75]).reset_index()
nyse_bm=nyse_bm[['jdate','25%','75%']].rename(columns={'25%':'bm25', '75%':'bm75'})

nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

# join back size and beme breakdown
ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])

# function to assign sz and bm bucket
def sz_bucket(row):
    if row['me']<=row['sz25']:
        value = 'S'
    elif row['me']<=row['sz75']:
        value='R'
    elif row['me']>row['sz75']:
        value='B'
    else:
        value=''    
    return value

def bm_bucket(row):
    if 0<=row['beme']<=row['bm25']:
        value = 'L'
    elif row['beme']<=row['bm75']:
        value='M'
    elif row['beme']>row['bm75']:
        value='H'
    else:
        value=''
    return value

# assign size portfolio
ccm1_jun['szport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(sz_bucket, axis=1), '')

# assign book-to-market portfolio
ccm1_jun['bmport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(bm_bucket, axis=1), '')

# create positivebmeme and nonmissport variable
ccm1_jun['posbm']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), 1, 0)
ccm1_jun['nonmissport']=np.where((ccm1_jun['bmport']!=''), 1, 0)

# store portfolio assignment as of June
june=ccm1_jun[['permno','date', 'jdate', 'bmport','szport','posbm','nonmissport']]
june['ffyear']=june['jdate'].dt.year
june['ffmonth'] = june['jdate'].dt.month
june['ffquarter'] = june['jdate'].dt.quarter

# merge back with monthly records
crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear', 'ffquarter', 'jdate']]
ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','ffmonth', 'ffquarter','szport','bmport','posbm','nonmissport']], how='left', on=['permno','ffyear', 'ffquarter'])

# keeping only records that meet the criteria
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]

#ccm12 = ccm2.drop_duplicates(["permno", "datadate"])
#ccm12['permno'] = ccm12['permno'].astype( int )
#ccm12['jdate'] = ccm2['datadate']+MonthEnd(0)
#bookvalue = ccm12.pivot( index='jdate', columns='permno', values='be' )

#marketcap = crsp2.pivot( index='jdate', columns='permno', values='me' )
#marketcap[ bookvalue.columns.to_list() ]
#shrcode = crsp2.pivot( index='jdate', columns='permno', values='shrcd' )
#exchcode = crsp2.pivot( index='jdate', columns='permno', values='exchcd' )
#ret = crsp2.pivot( index='jdate', columns='permno', values='ret' )
#retx = crsp2.pivot( index='jdate', columns='permno', values='retx' )
#retadj = crsp2.pivot( index='jdate', columns='permno', values='retadj' )3

############################
# Form Fama French Factors #
############################

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


# value-weigthed return
vwret=ccm4.groupby(['jdate','szport','bmport']).apply(wavg, 'retadj','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
vwret['sbport']=vwret['szport']+vwret['bmport']

# firm count
vwret_n=ccm4.groupby(['jdate','szport','bmport'])['retadj'].count().reset_index().rename(columns={'retadj':'n_firms'})
vwret_n['sbport']=vwret_n['szport']+vwret_n['bmport']

# tranpose
ff_factors=vwret.pivot(index='jdate', columns='sbport', values='vwret').reset_index()
ff_nfirms=vwret_n.pivot(index='jdate', columns='sbport', values='n_firms').reset_index()

# create SMB and HML factors
ff_factors[ 'WH' ]=( ff_factors['BH']+ff_factors['SH']+ ff_factors['RH'])/3
ff_factors['WL']=(ff_factors['BL']+ff_factors['SL'] + ff_factors['RL'])/3
ff_factors['WHML'] = ff_factors['WH']-ff_factors['WL']

ff_factors['WB']=(ff_factors['BL']+ff_factors['BM']+ff_factors['BH'])/3
ff_factors['WS']=(ff_factors['SL']+ff_factors['SM']+ff_factors['SH'])/3
ff_factors['WSMB'] = ff_factors['WS']-ff_factors['WB']
ff_factors=ff_factors.rename(columns={'jdate':'date'})

# n firm count
ff_nfirms['H']=ff_nfirms['SH']+ff_nfirms['BH'] + ff_nfirms['RH']
ff_nfirms['L']=ff_nfirms['SL']+ff_nfirms['BL'] + ff_nfirms['RL']

ff_nfirms['HML']=ff_nfirms['H']+ff_nfirms['L']

ff_nfirms['B']=ff_nfirms['BL']+ff_nfirms['BM']+ff_nfirms['BH']
ff_nfirms['S']=ff_nfirms['SL']+ff_nfirms['SM']+ff_nfirms['SH']
ff_nfirms['SMB']=ff_nfirms['B']+ff_nfirms['S']
ff_nfirms['TOTAL']=ff_nfirms['SMB']
ff_nfirms=ff_nfirms.rename(columns={'jdate':'date'})

###################
# Compare With FF #
###################
_ff = conn.get_table(library='ff', table='factors_monthly')
_ff=_ff[['date','mktrf','smb','hml', 'rf']]
_ff['date']=_ff['date']+MonthEnd( 0 )

_ffcomp = pd.merge(_ff, ff_factors[['date','WSMB','WHML']], how='inner', on=['date'])
_ffcomp70=_ffcomp[_ffcomp['date']>='01/01/1970']
_ffcomp70.set_index('date', inplace=True)
print(stats.pearsonr(_ffcomp70['smb'], _ffcomp70['WSMB']))
print(stats.pearsonr(_ffcomp70['hml'], _ffcomp70['WHML']))

pd.DataFrame(
    [ 
        stats.pearsonr(_ffcomp70['smb'], _ffcomp70['WSMB'])[0],
        stats.pearsonr(_ffcomp70['hml'], _ffcomp70['WHML'])[0]
    ], columns=['Correlation'], index=['SMB', 'HML']
).T.round( 4 ).to_latex( 'tables/corrFacteur.tex' )

_ffcomp.head(2)

(1+_ffcomp70[['hml', 'WHML']]).cumprod().plot()
(1+_ffcomp70[['smb', 'WSMB']]).cumprod().plot()

####################################################################
#################### Test the statistical model ####################
####################################################################

famaFrenchFactor = _ff
factorStar = ff_factors[ff_factors['date']>='1963-01-01']
factorStar.set_index('date', inplace=True)
allFactor = pd.merge(factorStar, famaFrenchFactor, how='inner', on='date')
allFactor.set_index( 'date', inplace=True )

# get some test assets
testassets = conn.get_table( library='ff', table='portfolios25', date_cols=['date'], index_col='date' )
testassets.index = testassets.index + MonthEnd( 0 )
testassets = testassets.iloc[ :, :25 ]

allData = pd.merge( allFactor, testassets, how='left', on='date' )
testassets = allData.iloc[ :, -25:]

# load industry portfolio
industries = pd.read_excel( 'FF10Industry.xlsx', index_col=0, parse_dates=True ) / 100
testassets = pd.merge(testassets, industries, how='inner', on='date')
tosave = testassets.subtract(allFactor[ 'rf' ], axis=0)
tosave.to_excel( 'testassets.xlsx' )

X1 = allFactor[ [ 'mktrf', 'smb', 'hml' ] ]
X1.to_excel( 'FFFactors.xlsx' )

X1Constant = sm.add_constant( X1 )
alphasFF = pd.DataFrame( [] )
marketFF = pd.DataFrame( [] )
smbFF = pd.DataFrame( [] )
hmlFF = pd.DataFrame( [] )
r2FF = pd.DataFrame( [] )
residualFF = pd.DataFrame( [] )

for testasset in testassets.columns:
    y = testassets[ testasset ]
    excessY = y - allFactor[ 'rf' ]

    mod = sm.OLS( excessY, X1Constant )
    res = mod.fit()
    alphasFF[ testasset ] = [ res.params[ 0 ],  res.tvalues[ 0 ] ]
    marketFF[ testasset ] = [ res.params[ 1 ],  res.tvalues[ 1 ] ]
    smbFF[ testasset ] = [ res.params[ 2 ],  res.tvalues[ 2 ] ]
    hmlFF[ testasset ] = [ res.params[ 3 ],  res.tvalues[ 3 ] ]
    r2FF[ testasset ] = [ res.rsquared ]
    residualFF[ testasset ] = res.resid

alphasFF.index = [ 'alpha', 'tstatAlpha' ]
marketFF.index = [ 'market', 'tstatBeta' ]
smbFF.index = [ 'smb', 'tstatSMB' ]
hmlFF.index = [ 'hml', 'tstatHML' ]
r2FF.index = ['R2']
FFresults = pd.concat([alphasFF, marketFF, smbFF, hmlFF, r2FF], axis=0)
print( FFresults )

colsname = ['LowBM', 'BM2', 'BM3', 'BM4', 'HighBM']
idxName = ['Small', 'ME2', 'ME3', 'ME4', 'Big']

aFF = pd.DataFrame( alphasFF.iloc[ 0, :25 ].values.reshape( [ 5, 5 ] ),
    columns=colsname, index=idxName )
aFF.round( 4 ).to_latex( 'tables/aFF.tex' )
aFFTstat = pd.DataFrame( alphasFF.iloc[ 1, :25 ].values.reshape( [ 5, 5 ] ),
    columns=colsname, index=idxName )
aFFTstat.round( 4 ).to_latex( 'tables/aFFTstat.tex' )

alphasFF.iloc[ :, 25: ].round( 4 ).to_latex( 'tables/aFFindustry.tex' )

muFF =  allFactor[ [ 'mktrf', 'smb', 'hml' ] ].mean()
covFF =  allFactor[ [ 'mktrf', 'smb', 'hml' ] ].cov()
covFFe = residualFF.cov()

T = residualFF.__len__()
N = residualFF.shape[ 1 ]
K = muFF.shape[ 0 ]

# tstat for the mean as in ff
averageFFTstats = ( ( T**( 1/2 ) )*muFF.values ) / ( np.diag( covFF.values )**( 1/2 ) )
print( averageFFTstats )

factorPart = 1 + ( muFF.values.T @ np.linalg.inv( covFF ) ) @ muFF.values 
alphapart = alphasFF.iloc[ 0, :25 ].values.T @ np.linalg.inv( covFFe.iloc[ :25, :25] ) @ alphasFF.iloc[ 0, :25 ].values
alphapartFull = alphasFF.loc[ 'alpha', : ].values.T @ np.linalg.inv( covFFe ) @ alphasFF.loc[ 'alpha', : ].values
jointTestAlphaFF = ( ( T - N - K) / N ) * ( factorPart ** ( -1 ) ) * alphapart
jointTestAlphaFFFull = ( ( T - N - K) / N ) * ( factorPart ** ( -1 ) ) * alphapartFull
cvJointAlphaTestFull = 1 - stats.f.ppf(0.95, T - N - K, N )
cvJointAlphaTest = 1 - stats.f.ppf(0.95,  T - ( N-10 ) - K, N-10 ) ##### need to check 
print( jointTestAlphaFF, cvJointAlphaTest, jointTestAlphaFFFull, cvJointAlphaTestFull )

## fama french star
X1 = allFactor[ [ 'mktrf', 'WSMB', 'WHML' ] ]
X1.to_excel( 'FFFactorsStar.xlsx' )

X1Constant = sm.add_constant( X1 )

alphasFFStar = pd.DataFrame( [] )
marketFFStar = pd.DataFrame( [] )
smbFFStar = pd.DataFrame( [] )
hmlFFStar = pd.DataFrame( [] )
r2FFStar = pd.DataFrame( [] )
residualFFStar = pd.DataFrame( [] )

for testasset in testassets.columns:
    y = testassets[ testasset ]
    excessY = y - allFactor[ 'rf' ]

    mod = sm.OLS( excessY, X1Constant )
    res = mod.fit()
    alphasFFStar[ testasset ] = [ res.params[ 0 ],  res.tvalues[ 0 ] ]
    marketFFStar[ testasset ] = [ res.params[ 1 ],  res.tvalues[ 1 ] ]
    smbFFStar[ testasset ] = [ res.params[ 2 ],  res.tvalues[ 2 ] ]
    hmlFFStar[ testasset ] = [ res.params[ 3 ],  res.tvalues[ 3 ] ]
    r2FFStar[ testasset ] = [ res.rsquared ]
    residualFFStar[ testasset ] = res.resid

alphasFFStar.index = [ 'alpha', 'tstatAlpha' ]
marketFFStar.index = [ 'market', 'tstatBeta' ]
smbFF.index = [ 'smb', 'tstatSMB' ]
hmlFFStar.index = [ 'hml', 'tstatHML' ]
r2FFStar.index = ['R2']
FFresultsStar = pd.concat( [ alphasFFStar, marketFFStar, smbFFStar,hmlFFStar, r2FFStar ],
                             axis=0 )
print( FFresultsStar )

aFFSTar = pd.DataFrame( alphasFFStar.iloc[ 0, :25 ].values.reshape( [ 5, 5 ] ),
    columns=colsname, index=idxName )
aFFSTar.round( 4 ).to_latex( 'tables/aFFStar.tex', )
aFFStarTstat = pd.DataFrame( alphasFFStar.iloc[ 1, :25 ].values.reshape( [ 5, 5 ] ),
    columns=colsname, index=idxName )
aFFStarTstat.round( 4 ).to_latex( 'tables/aFFSTarTstat.tex', )

alphasFFStar.iloc[ :, 25: ].round( 4 ).to_latex( 'tables/aFFStarindustry.tex' )

muFFStar =  allFactor[ [ 'mktrf', 'WSMB', 'WHML' ] ].mean()
covFFStar =  allFactor[ [ 'mktrf', 'WSMB', 'WHML' ] ].cov()
covFFeStar = residualFFStar.cov()

averageFFStarTstats = ( ( T**( 1/2 ) )*muFFStar.values ) / ( np.diag( covFFStar.values )**( 1/2 ) )
print( averageFFStarTstats )

factorPartStar = 1 + ( muFFStar.values @ np.linalg.inv( covFFStar ) ) @ muFFStar.values 
alphapartStarFull = alphasFFStar.loc[ 'alpha', : ].values.T @ np.linalg.inv( covFFeStar ) @ alphasFFStar.loc[ 'alpha', : ].values
alphapartStar = alphasFFStar.iloc[ 0, :25 ].values.T @ np.linalg.inv( covFFeStar.iloc[ :25, :25] ) @ alphasFFStar.iloc[ 0, :25 ].values
jointTestAlphaFFStar = ( ( T - N - K ) / N ) * ( factorPartStar ** ( -1 ) ) * alphapartStar
jointTestAlphaFFStarFull = ( ( T - N - K ) / N ) * ( factorPartStar ** ( -1 ) ) * alphapartStarFull
1 - stats.f.cdf([ jointTestAlphaFFStar, jointTestAlphaFFStarFull ])
print( jointTestAlphaFFStar, 1 - stats.f.cdf( jointTestAlphaFFStar, ( N - 10 ), ( T - ( N - 10 ) - K ) ) )
print( jointTestAlphaFFStarFull, 1 - stats.f.cdf( jointTestAlphaFFStarFull, N, ( T - N - K ) ) )

grsTest = pd.DataFrame( np.vstack(
    [ np.hstack( [jointTestAlphaFF, jointTestAlphaFFFull, jointTestAlphaFFStar, jointTestAlphaFFStarFull]),
     np.hstack( [ 1 - stats.f.cdf( jointTestAlphaFFStar, ( N - 10 ), ( T - ( N - 10 ) - K ) ),
      1 - stats.f.cdf( jointTestAlphaFFStarFull, N, ( T - N - K ) ),
       1 - stats.f.cdf( jointTestAlphaFFStar, ( N - 10 ), ( T - ( N - 10 ) - K ) ),
       1 - stats.f.cdf( jointTestAlphaFFStarFull, N, ( T - N - K ) ) ])] ),
      columns=[ 'FF3 25x25', 'FF3 25x25+industry', 'FF3^* 25x25', 'FF3^* 25x25+industry'], index=['GRS', 'p-Values'])
grsTest.round( 4 ).to_latex( 'tables/grs.tex' )

#############3 Sharpe ratio test
FA = allFactor[ [ 'mktrf', 'smb', 'hml' ] ].values
FB = allFactor[ [ 'mktrf', 'WSMB', 'WHML' ] ].values
KA = K
KB = K
muA = muFF.values.reshape( [ -1, 1 ] )
muB = muFFStar.values.reshape( [ -1, 1 ] )
VA = covFF.values
VB = covFFStar.values
WA = ( ( T - K - 2 ) / T )*np.linalg.inv( VA )
WB = ( ( T - K - 2 ) / T )*np.linalg.inv( VB )

theta2A = ( muA.T @ WA ) @ muA - ( KA / T ) 
theta2B = ( muB.T @ WB ) @ muB - ( KB / T )

dtheta2 = np.abs( theta2A - theta2B ) 

FAd = FA - np.ones( ( T, 1 ) ) @ muA.T
FBd = FB - np.ones( ( T, 1 ) ) @ muB.T
uA = FAd @ WA @ muA
uB = FBd @ WB @ muB
dt = 2*( uA - uB ) - ( uA**2 -uB**2 ) # imposing the null
vd = nw( dt, lag=0 )
dt1 = 2*( uA-uB ) - ( uA**2 - uB**2 ) + ( theta2A - theta2B ) # % not imposing the null
vd1 = nw( dt1, lag=0 )
pval2a = 2 * ( 1-stats.norm.cdf( np.abs( dtheta2 ) / np.sqrt( vd1 / T ) ) )
#pval2b = 2 * (1-stats.norm.cdf( np.abs( dtheta2 ) / np.sqrt( vd / T ) ) )
pd.DataFrame( 
    [
        dtheta2[0][0], pval2a[0][0]
    ],
    columns=['SR^2 test'], index=['tstat', 'pvalue']
).round(4).to_latex('tables/shr2test.tex')


