# Empirical Asset Pricing : HW1
# Authors : Maxime Borel 

import pandas as pd
import numpy as np
import wrds
import datetime as dt
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats


# established connection with the server
db = wrds.Connection( wrds_username="maxrel95" )
db.create_pgpass_file()

comp = db.raw_sql("""

                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk 
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1961'
                    """,
                    date_cols = [ 'datadate' ]
)

comp[ 'year' ] = comp[ 'datadate' ].dt.year

# create preferrerd stock
# put in ps reedm val of prefered if nul replace liquididating 
#if still null, replace by current prefered stock else 0
comp[ 'ps' ] = np.where( comp[ 'pstkrv' ].isnull(), comp[ 'pstkl' ], comp[ 'pstkrv' ] )
comp['ps'] = np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(),0,comp['ps'])
comp['txditc'] = comp['txditc'].fillna(0)

# create book equity
# book value is shareholder equity +differed taxc and inv tax - preferes stock
comp['be'] = comp['seq']+comp['txditc']-comp['ps']
comp['be'] = np.where(comp['be']>0, comp['be'], np.nan) # remove firm with negative bv

comp = comp.sort_values(by = ['gvkey','datadate']) # sort dataset by id and date
comp['count'] = comp.groupby(['gvkey']).cumcount() # cumsum of appearanc

comp = comp[['gvkey','datadate','year','be','count']]

# get price data
# two tables, crsp.msf, crsp.msenames, keep only firm in crsp.msf
# match date, exchcd  =  NYSE + NASDAQ
crsp_m  =  db.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1961' and '12/31/2022'
                      and b.exchcd between 1 and 3
                      """, date_cols = ['date']) 

# columns as int type
crsp_m[['permco','permno','shrcd','exchcd']] = crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Line up date to be end of month in a new col 
crsp_m['jdate'] = crsp_m['date']+MonthEnd(0)

# add delisting return
#table monthly stock event delisting
dlret = db.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """, date_cols = ['dlstdt'])

# do th same for dlret for matching
dlret.permno = dlret.permno.astype(int)
dlret['jdate'] = dlret['dlstdt']+MonthEnd(0)

# merged the two table in a table 
crsp = pd.merge(crsp_m, dlret, how = 'left',on = ['permno','jdate'])
crsp['dlret'] = crsp['dlret'].fillna(0) # fill nan with zero in dlret 
crsp['ret'] = crsp['ret'].fillna(0) # fill missing ret by 0

# adjust ret to deleting ret
crsp['retadj'] = (1+crsp['ret'])*(1+crsp['dlret'])-1

# calculate market equity price*shareoutstanding
crsp['me'] = crsp['prc'].abs()*crsp['shrout'] 
crsp = crsp.drop(['dlret','dlstdt','prc','shrout'], axis = 1) # remove unecessary var
crsp = crsp.sort_values(by = ['jdate','permco','me']) # sort by date, id, and size 

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()

# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate','permco'])['me'].max().reset_index()

# join by jdate/maxme to find the permno, keep only the permno of the largest firm 
crsp1 = pd.merge(crsp, crsp_maxme, how = 'inner', on = ['jdate','permco','me'])

# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis = 1)

# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how = 'inner', on = ['jdate','permco'])

# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by = ['permno','jdate']).drop_duplicates()

# keep December market cap
crsp2['year'] = crsp2['jdate'].dt.year
crsp2['month'] = crsp2['jdate'].dt.month
qtrme = crsp2[ crsp2['month'].isin( [ 3, 6, 9, 12 ] ) ]
qtrme = qtrme[['permno','date','jdate','me','year', 'month']].rename(columns = {'me':'qtr_me'})

crsp2['ffdate'] = crsp2['jdate']
crsp2['ffyear'] = crsp2['ffdate'].dt.year
crsp2['ffmonth'] = crsp2['ffdate'].dt.month
crsp2['ffquarter']  =  ( crsp2['ffmonth']-1 )//3 + 1
crsp2[ 'ffadaptedMonth' ] = crsp2.groupby( ['permno', 'ffyear', 'ffquarter']).cumcount()
crsp2[ 'ffadaptedMonth' ] = crsp2[ 'ffadaptedMonth' ] + 1
crsp2['1+retx'] = 1+crsp2['retx']
crsp2 = crsp2.sort_values(by = ['permno','date'])

# cumret by stock
crsp2['cumretx'] = crsp2.groupby(['permno','ffyear', 'ffquarter'])['1+retx'].cumprod()

crsp2['lcumretx'] = crsp2.groupby(['permno'])['cumretx'].shift(1)

# lag market cap
crsp2['lme'] = crsp2.groupby(['permno'])['me'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
crsp2['count'] = crsp2.groupby(['permno']).cumcount()
crsp2['lme'] = np.where(crsp2['count'] ==0, crsp2['me']/crsp2['1+retx'], crsp2['me'])

# baseline me
mebase=crsp2[crsp2['ffadaptedMonth']==1][['permno','ffyear', 'ffquarter', 'lme']].rename(columns={'lme':'mebase'})

# merge result back together
crsp3=pd.merge(crsp2, mebase, how='left', on=['permno','ffyear', 'ffquarter'])
crsp3['wt']=np.where(crsp3['ffadaptedMonth']==1, crsp3['lme'], crsp3['mebase']*crsp3['cumretx'])

#decme['year']=decme['year']+1
qtrme=qtrme[['permno','year', 'month', 'qtr_me']]

# Info as of June
crsp3_qtr = crsp3[crsp3[ 'month' ].isin( [ 3, 6, 9, 12 ] )]

crsp3_qtr = pd.merge(crsp3_qtr, qtrme, how='inner', on=['permno','year', 'month'])
crsp3_qtr = crsp3_qtr[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','qtr_me']]
crsp3_qtr = crsp3_qtr.sort_values(by=['permno','jdate']).drop_duplicates()

#######################
# CCM Block           #
#######################
ccm = db.raw_sql("""
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
ccm_jun=pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['beme']=ccm_jun['be']*1000/ccm_jun['dec_me']

# select NYSE stocks for bucket breakdown
# exchcd = 1 and positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
nyse=ccm_jun[(ccm_jun['exchcd']==1) & (ccm_jun['beme']>0) & (ccm_jun['me']>0) & \
             (ccm_jun['count']>=1) & ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))]

# size breakdown
nyse_sz=nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me':'sizemedn'})

# beme breakdown
nyse_bm=nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_bm=nyse_bm[['jdate','30%','70%']].rename(columns={'30%':'bm30', '70%':'bm70'})

nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

# join back size and beme breakdown
ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])

# function to assign sz and bm bucket
def sz_bucket(row):
    if row['me']==np.nan:
        value=''
    elif row['me']<=row['sizemedn']:
        value='S'
    else:
        value='B'
    return value

def bm_bucket(row):
    if 0<=row['beme']<=row['bm30']:
        value = 'L'
    elif row['beme']<=row['bm70']:
        value='M'
    elif row['beme']>row['bm70']:
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

# merge back with monthly records
crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear','jdate']]
ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','szport','bmport','posbm','nonmissport']], how='left', on=['permno','ffyear'])

# keeping only records that meet the criteria
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]


