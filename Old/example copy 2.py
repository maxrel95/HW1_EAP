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
                      select a.permno, a.permco, a.mthcaldt, 
                      a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg, 
                      a.primaryexch, a.conditionaltype, a.tradingstatusflg,
                      a.mthret, a.mthretx, a.shrout, a.mthprc
                      from crsp.msf_v2 as a
                      where a.mthcaldt between '01/01/1959' and '12/31/2021'
                      """, date_cols=['mthcaldt']) 

crsp_m = crsp_m.loc[(crsp_m.sharetype=='NS') & \
                    (crsp_m.securitytype=='EQTY') & \
                    (crsp_m.securitysubtype=='COM') & \
                    (crsp_m.usincflg=='Y') & \
                    (crsp_m.issuertype.isin(['ACOR', 'CORP']))]

crsp_m = crsp_m.loc[(crsp_m.primaryexch.isin(['N', 'A', 'Q'])) & \
                   (crsp_m.conditionaltype =='RW') & \
                   (crsp_m.tradingstatusflg =='A')]

# change variable format to int
crsp_m[['permco','permno']]=crsp_m[['permco','permno']].astype(int)

# Line up date to be end of month
crsp_m['jdate']=crsp_m['mthcaldt']+MonthEnd(0)

crsp = crsp_m.copy()

crsp['mthret']=crsp['mthret'].fillna(0)
crsp['mthretx']=crsp['mthretx'].fillna(0)
crsp['me']=crsp['mthprc']*crsp['shrout'] 
crsp=crsp.drop(['mthprc','shrout'], axis=1)
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
decme=decme[['permno','mthcaldt','jdate','me','year', 'month', 'quarter']].rename(columns={'me':'dec_me'}) # lagged by 6 month the market cap
decme['ffdate'] = decme['jdate']+MonthEnd(6)
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
crsp2['1+retx']=1+crsp2['mthretx']
crsp2=crsp2.sort_values(by=['permno','mthcaldt'])

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
mebase=crsp2[crsp2['ffmonthadapted']==1][ [ 'permno','ffyear', 'ffquarter', 'lme' ] ].rename(columns={'lme':'mebase'})

# merge result back together
crsp3=pd.merge(crsp2, mebase, how='left', on=['permno','ffyear', 'ffquarter'])
crsp3['wt']=np.where(crsp3['ffmonthadapted']==1, crsp3['lme'], crsp3['mebase']*crsp3['lcumretx'])


# Info as of June
#crsp3_jun = crsp3[crsp3['ffmonthadapted']==1 ]
crsp3_jun = crsp3[crsp3['ffmonthadapted']==3 ]


crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno','ffyear', 'ffquarter'])
crsp_jun=crsp_jun[['permno','mthcaldt', 'jdate', 'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype', \
                   'primaryexch', 'conditionaltype', 'tradingstatusflg', \
                   'mthret','me','wt','cumretx','mebase','lme','dec_me']]
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
ccm1['jdate']=ccm1['yearend']+MonthEnd(6) # try 3

# set link date bounds
ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
ccm2=ccm2[['gvkey','permno','datadate','yearend', 'jdate','be', 'count']]

# link comp and crsp
ccm_jun=pd.merge_ordered(crsp_jun, ccm2, how='left', on=['permno', 'jdate'], fill_method='ffill')
ccm_jun['beme']=ccm_jun['be']*1000/ccm_jun['dec_me']

# positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
#(ccm_jun['primaryexch']=='N') &
nyse=ccm_jun[ (ccm_jun['beme']>0) & (ccm_jun['me']>0) & \
             (ccm_jun['count']>=1)]
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
june=ccm1_jun[['permno','mthcaldt', 'jdate', 'bmport','szport','posbm','nonmissport']]
june['ffyear']=june['jdate'].dt.year
june['ffmonth'] = june['jdate'].dt.month
june['ffquarter'] = june['jdate'].dt.quarter

# merge back with monthly records
crsp3 = crsp3[['mthcaldt','permno', 'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype', \
               'primaryexch', 'conditionaltype', 'tradingstatusflg', \
               'mthret', 'me','wt','cumretx','ffyear', 'ffquarter', 'jdate']]
ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','ffmonth', 'ffquarter','szport','bmport','posbm','nonmissport']], how='left', on=['permno','ffyear', 'ffquarter'])

# keeping only records that meet the criteria
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) ]


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
vwret=ccm4.groupby(['jdate','szport','bmport']).apply(wavg, 'mthret','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
vwret['sbport']=vwret['szport']+vwret['bmport']

# firm count
vwret_n=ccm4.groupby(['jdate','szport','bmport'])['mthret'].count().reset_index().rename(columns={'mthret':'n_firms'})
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
_ff['date']=_ff['date']+MonthEnd(0)

_ffcomp = pd.merge(_ff, ff_factors[['date','WSMB','WHML']], how='inner', on=['date'])
_ffcomp70=_ffcomp[_ffcomp['date']>='01/01/1970']
_ffcomp70.set_index('date', inplace=True)
print(stats.pearsonr(_ffcomp70['smb'], _ffcomp70['WSMB']))
print(stats.pearsonr(_ffcomp70['hml'], _ffcomp70['WHML']))

_ffcomp.head(2)

(1+_ffcomp70[['hml', 'WHML']]).cumprod().plot()
(1+_ffcomp70[['smb', 'WSMB']]).cumprod().plot()

famaFrenchFactor = _ff
factorStar = ff_factors[ff_factors['date']>='1963-01-01']
factorStar.set_index('date', inplace=True)
allFactor = pd.merge(factorStar, famaFrenchFactor, how='inner', on='date')
allFactor.set_index( 'date', inplace=True )

#############################################################
#############################################################
#############################################################

testassets = conn.get_table( library='ff', table='portfolios25', date_cols=['date'], index_col='date' )
testassets.index = testassets.index + MonthEnd( 0 )

allData = pd.merge( allFactor, testassets, how='left', on='date' )
testassets = allData.iloc[ :, -25:]

## need to do some regression and keep the alpha













