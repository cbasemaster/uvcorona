# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import interpolate

def tangent(x,y,a,glob=False):
 # interpolate the data with a spline
 spl = interpolate.splrep(x,y)
 if glob==False:
     small_t = np.arange(a-2,a+2)
 else:
     small_t = np.arange(a-2,len(x)+2)
 fa = interpolate.splev(a,spl,der=0)     # f(a)
 fprime = interpolate.splev(a,spl,der=1) # f'(a)
 #print fprime
 tan = fa+fprime*(small_t-a) # tangent
 return fprime
 #plt.plot(a,fa,'om',small_t,tan,'--r')


#matplotlib inline
sns.set()
df = pd.read_csv('data/time-series-19-covid-combined.csv', skiprows=1)
df.head()
df.info()

df.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']
df.head
#print df.day
df.fillna
df.day = pd.to_datetime(df.day)
df.set_index('day', inplace=True)
is_indo =  df['country']=='Indonesia'
is_malay =  df['country']=='Malaysia'
is_thai =  df['country']=='Thailand'
is_japan =  df['country']=='Japan'
is_india =  df['country']=='India'
is_saudi =  df['country']=='Saudi Arabia'
is_brazil =  df['country']=='Brazil'
is_egypt =  df['country']=='Egypt'
is_sin =  df['country']=='Singapore'
is_russia =  df['country']=='Russia'
is_france =  df['country']=='France'

is_australia =  (df['country']=='Australia') & (df['territory']=='South Australia')
is_west_australia =  (df['country']=='Australia') & (df['territory']=='Western Australia')
is_south_africa =  df['country']=='South Africa'
is_argentina =  df['country']=='Argentina'
is_nz =  df['country']=='News Zealand'
is_botswana =  df['country']=='Botswana'

is_us =  df['country']=='US'
is_italy =  df['country']=='Italy'
is_belgium =  (df['country']=='Belgium')
is_nl =  (df['country']=='Netherlands') & (df['lat']==52.1326)
is_korea =  df['country']=='Korea, South'
is_canada =  (df['country']=='Canada') & (df['territory']=='British Columbia')
is_iran =  df['country']=='Iran'
is_uk =  (df['country']=='UK') & (df['territory']=='')
is_china =  (df['country']=='China')
is_china_tianjin =  (df['country']=='China') & (df['territory']=='Tianjin')
is_china_hebei =  (df['country']=='China') & (df['territory']=='Hebei')
is_china_qinghai =  (df['country']=='China') & (df['territory']=='Qinghai')
is_china_liaoning = (df['country']=='China') & (df['territory']=='Liaoning')
#is_russia =  df['country']=='Indonesia'
chn_region= df[(is_china)]['territory'].unique()
marker = itertools.cycle((',',',', '+', '+', '.','.', 'o', '*')) 
#print chn_region

#print df[(is_france)]['territory'].unique()
#ax0=plt.figure(figsize=(15,10))
fig, ax0 = plt.subplots(figsize=(15,10))
#ax0.plot(df[(is_malay)][['confirmed']],color='green', label='Malaysia')
ax0.plot(df[(is_indo)][['confirmed']],color='green', label='Indonesia')
ax0.plot(df[(is_thai)][['confirmed']],color='green', label='Thailand')
ax0.plot(df[(is_egypt)][['confirmed']],color='green', label='Egypt')
ax0.plot(df[(is_india)][['confirmed']],color='green', label='India')
ax0.plot(df[(is_saudi)][['confirmed']],color='green', label='Saudi Arabia')
ax0.plot(df[(is_brazil)][['confirmed']],color='green', label='Brazil')

ax0.plot(df[(is_australia)][['confirmed']],color='red', label='South Australia')
ax0.plot(df[(is_west_australia)][['confirmed']],color='red', label='Western Australia')
ax0.plot(df[(is_nz)][['confirmed']],color='red', label='New Zealand')
ax0.plot(df[(is_argentina)][['confirmed']],color='red', label='Argentina')
ax0.plot(df[(is_south_africa)][['confirmed']],color='red', label='South Africa')
ax0.plot(df[(is_botswana)][['confirmed']],color='red', label='Botswana')

#ax0.plot(df[(is_iran)][['confirmed']],color='blue', label='Iran')
ax0.plot(df[(is_nl)][['confirmed']],color='blue', label='Netherlands')
ax0.plot(df[(is_belgium)][['confirmed']],color='blue', label='Belgium')
ax0.plot(df[(is_japan)][['confirmed']],color='blue', label='Japan')
ax0.plot(df[(is_korea)][['confirmed']],color='blue', label='South Korea')
ax0.plot(df[(is_canada)][['confirmed']],color='blue', label='Canada')
ax0.plot(df[(is_russia)][['confirmed']],color='blue', label='Russia')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
plt.savefig('data/season.png')





ax=plt.figure(figsize=(15,10))
plt.plot(df[(is_malay)][['confirmed']], label='Malaysia')
plt.plot(df[(is_indo)][['confirmed']], label='Indonesia')
plt.plot(df[(is_thai)][['confirmed']], label='Thailand')
plt.plot(df[(is_japan)][['confirmed']], label='Japan')
plt.plot(df[(is_india)][['confirmed']], label='India')
plt.plot(df[(is_russia)][['confirmed']], label='Russia')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/asia.png')

ax2=plt.figure(figsize=(15,10))
#plt.plot(df[(is_china)][['confirmed']], label='China')
plt.plot(df[(is_us)][['confirmed']], label='US')
plt.plot(df[(is_italy)][['confirmed']], label='Italy')
plt.plot(df[(is_france)][['confirmed']], label='France')
plt.plot(df[(is_nl)][['confirmed']], label='Netherlands')
plt.plot(df[(is_iran)][['confirmed']], label='Iran')
plt.plot(df[(is_indo)][['confirmed']], label='Indonesia')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/europe.png')

ax3=plt.figure(figsize=(15,10))
plt.plot(df[(is_malay)][['confirmed']].diff(), label='Malaysia')
plt.plot(df[(is_indo)][['confirmed']].diff(), label='Indonesia')
plt.plot(df[(is_thai)][['confirmed']].diff(), label='Thailand')
plt.plot(df[(is_japan)][['confirmed']].diff(), label='Japan')
plt.plot(df[(is_india)][['confirmed']].diff(), label='India')
plt.plot(df[(is_russia)][['confirmed']].diff(), label='Russia')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/asia_diff.png')

ax4=plt.figure(figsize=(15,10))
plt.plot(df[(is_us)][['confirmed']].diff(), label='US')
plt.plot(df[(is_italy)][['confirmed']].diff(), label='Italy')
plt.plot(df[(is_france)][['confirmed']].diff(), label='France')
plt.plot(df[(is_nl)][['confirmed']].diff(), label='Netherlands')
plt.plot(df[(is_iran)][['confirmed']].diff(), label='Iran')
plt.plot(df[(is_indo)][['confirmed']].diff(), label='Indonesia')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);

ax5=plt.figure(figsize=(15,10))
for chn in chn_region:
    #print chn
    #print df[(is_china)&(df['territory']==chn)][['confirmed']]
    plt.plot(df[(is_china)&(df['territory']==chn)][['confirmed']], label=chn,marker = marker.next())
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/china.png')

ax6=plt.figure(figsize=(15,10))
for chn in chn_region:
    #print chn
    #print df[(is_china)&(df['territory']==chn)][['confirmed']]
    plt.plot(df[(is_china)&(df['territory']==chn)][['confirmed']].diff(), label=chn,marker = marker.next())
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/china_diff.png')

#sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', fit_reg=False, data=df_iris, hue='target');


us=df[(is_us)][['confirmed','recovered','deaths']]
it=df[(is_italy)][['confirmed','recovered','deaths']]
thai=df[(is_thai)][['confirmed','recovered','deaths']]
japan=df[(is_japan)][['confirmed','recovered','deaths']]
india=df[(is_india)][['confirmed','recovered','deaths']]
australia=df[(is_australia)][['confirmed','recovered','deaths']]
indo=df[(is_indo)][['confirmed','recovered','deaths']]

tianjin=df[(is_china_tianjin)][['confirmed','recovered','deaths']]
hebei=df[(is_china_hebei)][['confirmed','recovered','deaths']]
liaoning=df[(is_china_liaoning)][['confirmed','recovered','deaths']]
qinghai=df[(is_china_qinghai)][['confirmed','recovered','deaths']]


from scipy import stats


us['no'] = np.arange(len(us))
us['local slope_confirmed'] = np.arange(len(us))
for i in np.arange(len(us)):
    slope=tangent(us['no'], us['confirmed'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    us['local slope_confirmed'][i] = slope
us['global slope_confirmed']=tangent(us['no'], us['confirmed'],i,glob=True)
del us['no']

it['no'] = np.arange(len(it))
it['local slope_confirmed'] = np.arange(len(it))
for i in np.arange(len(it)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(it['no'][0:i+1], it['confirmed'][0:i+1]) 
    slope=tangent(it['no'], it['confirmed'],i)
    it['local slope_confirmed'][i] =slope
it['global slope_confirmed']=tangent(it['no'], it['confirmed'],i,glob=True)
del it['no']

thai['no'] = np.arange(len(thai))
thai['local slope_confirmed'] = np.arange(len(thai))
for i in np.arange(len(thai)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(thai['no'][0:i+1], thai['confirmed'][0:i+1]) 
    slope=tangent(thai['no'], thai['confirmed'],i)
    thai['local slope_confirmed'][i] =slope
thai['global slope_confirmed']=tangent(thai['no'], thai['confirmed'],i,glob=True)
del thai['no']

japan['no'] = np.arange(len(japan))
japan['local slope_confirmed'] = np.arange(len(japan))
for i in np.arange(len(japan)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(japan['no'][0:i+1], japan['confirmed'][0:i+1])
    slope=tangent(japan['no'], japan['confirmed'],i)
    japan['local slope_confirmed'][i] =slope
japan['global slope_confirmed']=tangent(japan['no'], japan['confirmed'],i,glob=True)
del japan['no']

india['no'] = np.arange(len(india))
india['local slope_confirmed'] = np.arange(len(india))
for i in np.arange(len(india)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(india['no'][0:i+1], india['confirmed'][0:i+1])
    slope=tangent(india['no'], india['confirmed'],i)
    india['local slope_confirmed'][i] =slope
india['global slope_confirmed']=tangent(india['no'], india['confirmed'],i,glob=True)
del india['no']

australia['no'] = np.arange(len(australia))
australia['local slope_confirmed'] = np.arange(len(australia))
for i in np.arange(len(australia)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(australia['no'][0:i+1], australia['confirmed'][0:i+1]) 
    slope=tangent(australia['no'], australia['confirmed'],i)
    australia['local slope_confirmed'][i] =slope
australia['global slope_confirmed']=tangent(australia['no'], australia['confirmed'],i,glob=True)
del australia['no']

tianjin['no'] = np.arange(len(tianjin))
tianjin['local slope_confirmed'] = np.arange(len(tianjin))
for i in np.arange(len(tianjin)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(tianjin['no'][0:i+1], tianjin['confirmed'][0:i+1]) 
    slope=tangent(tianjin['no'], tianjin['confirmed'],i)
    tianjin['local slope_confirmed'][i] =slope
tianjin['global slope_confirmed']=tangent(tianjin['no'], tianjin['confirmed'],i,glob=True)
del tianjin['no']


hebei['no'] = np.arange(len(hebei))
hebei['local slope_confirmed'] = np.arange(len(hebei))
for i in np.arange(len(hebei)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(hebei['no'][0:i+1], hebei['confirmed'][0:i+1]) 
    slope=tangent(hebei['no'], hebei['confirmed'],i)
    hebei['local slope_confirmed'][i] =slope
hebei['global slope_confirmed']=tangent(hebei['no'], hebei['confirmed'],i,glob=True)
del hebei['no']


qinghai['no'] = np.arange(len(qinghai))
qinghai['local slope_confirmed'] = np.arange(len(qinghai))
for i in np.arange(len(qinghai)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(qinghai['no'][0:i+1], qinghai['confirmed'][0:i+1]) 
    slope=tangent(qinghai['no'], qinghai['confirmed'],i)
    qinghai['local slope_confirmed'][i] =slope
qinghai['global slope_confirmed']=tangent(qinghai['no'], qinghai['confirmed'],i,glob=True)
del qinghai['no']


liaoning['no'] = np.arange(len(liaoning))
liaoning['local slope_confirmed'] = np.arange(len(liaoning))
for i in np.arange(len(liaoning)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(liaoning['no'][0:i+1], liaoning['confirmed'][0:i+1]) 
    slope=tangent(liaoning['no'], liaoning['confirmed'],i)
    liaoning['local slope_confirmed'][i] =slope
liaoning['global slope_confirmed']=tangent(liaoning['no'], liaoning['confirmed'],i,glob=True)
del liaoning['no']

indo['no'] = np.arange(len(indo))
indo['slope'] = np.arange(len(indo))
for i in np.arange(len(indo)):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(liaoning['no'][0:i+1], liaoning['confirmed'][0:i+1]) 
    slope=tangent(indo['no'], indo['confirmed'],i)
    indo['slope'][i] =slope
indo['long_slope']=tangent(indo['no'], indo['confirmed'],i,glob=True)



us['confirmed']= us.confirmed.diff()
it['confirmed']= it.confirmed.diff()
thai['confirmed']=thai.confirmed.diff()
japan['confirmed']=japan.confirmed.diff()
india['confirmed']=india.confirmed.diff()
australia['confirmed']=australia.confirmed.diff()


tianjin['confirmed']=tianjin.confirmed.diff()
hebei['confirmed']=hebei.confirmed.diff()
liaoning['confirmed']=liaoning.confirmed.diff()
qinghai['confirmed']=qinghai.confirmed.diff()



us['recovered']= us.recovered.diff()
it['recovered']= it.recovered.diff()
thai['recovered']=thai.recovered.diff()
japan['recovered']=japan.recovered.diff()
india['recovered']=india.recovered.diff()
australia['recovered']=australia.recovered.diff()

tianjin['recovered']=tianjin.recovered.diff()
hebei['recovered']=hebei.recovered.diff()
liaoning['recovered']=liaoning.recovered.diff()
qinghai['recovered']=qinghai.recovered.diff()

us['deaths']= us.deaths.diff()
it['deaths']= it.deaths.diff()
thai['deaths']=thai.deaths.diff()
japan['deaths']=japan.deaths.diff()
india['deaths']=india.deaths.diff()
australia['deaths']=australia.deaths.diff()

tianjin['deaths']=tianjin.deaths.diff()
hebei['deaths']=hebei.deaths.diff()
liaoning['deaths']=liaoning.deaths.diff()
qinghai['deaths']=qinghai.deaths.diff()

frames = [us,it,thai,japan,india,australia]
result = pd.concat(frames).fillna(0)

frames_china = [tianjin, hebei, qinghai,liaoning]
result_china = pd.concat(frames_china).fillna(0)

us_uv=df1[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
it_uv=df5[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
thai_uv=df4[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
japan_uv=df6[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
india_uv=df7[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
australia_uv=df2[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]


us_uv['no'] = np.arange(len(us_uv))
us_uv['local slope_uv'] = np.arange(len(us_uv))
for i in np.arange(len(us_uv)):
    slope=tangent(us_uv['no'], us_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    us_uv['local slope_uv'][i] = slope
us_uv['global slope_uv']=tangent(us_uv['no'], us_uv['UVIEF'],i,glob=True)
us_uv['average_uv_index']=us_uv['UVIEF'].mean()
us_uv['average_ozone']=us_uv['ozone'].mean()
del us_uv['no']

it_uv['no'] = np.arange(len(it_uv))
it_uv['local slope_uv'] = np.arange(len(it_uv))
for i in np.arange(len(it_uv)):
    slope=tangent(it_uv['no'], it_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    it_uv['local slope_uv'][i] = slope
it_uv['global slope_uv']=tangent(it_uv['no'], it_uv['UVIEF'],i,glob=True)
it_uv['average_uv_index']=it_uv['UVIEF'].mean()
it_uv['average_ozone']=it_uv['ozone'].mean()
del it_uv['no']

thai_uv['no'] = np.arange(len(thai_uv))
thai_uv['local slope_uv'] = np.arange(len(thai_uv))
for i in np.arange(len(thai_uv)):
    slope=tangent(thai_uv['no'], thai_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    thai_uv['local slope_uv'][i] = slope
thai_uv['global slope_uv']=tangent(thai_uv['no'], thai_uv['UVIEF'],i,glob=True)
thai_uv['average_uv_index']=thai_uv['UVIEF'].mean()
thai_uv['average_ozone']=thai_uv['ozone'].mean()
del thai_uv['no']

japan_uv['no'] = np.arange(len(japan_uv))
japan_uv['local slope_uv'] = np.arange(len(japan_uv))
for i in np.arange(len(japan_uv)):
    slope=tangent(japan_uv['no'], japan_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    japan_uv['local slope_uv'][i] = slope
japan_uv['global slope_uv']=tangent(japan_uv['no'], japan_uv['UVIEF'],i,glob=True)
japan_uv['average_uv_index']=japan_uv['UVIEF'].mean()
japan_uv['average_ozone']=japan_uv['ozone'].mean()
del japan_uv['no']

india_uv['no'] = np.arange(len(india_uv))
india_uv['local slope_uv'] = np.arange(len(india_uv))
for i in np.arange(len(india_uv)):
    slope=tangent(india_uv['no'], india_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    india_uv['local slope_uv'][i] = slope
india_uv['global slope_uv']=tangent(india_uv['no'], india_uv['UVIEF'],i,glob=True)
india_uv['average_uv_index']=india_uv['UVIEF'].mean()
india_uv['average_ozone']=india_uv['ozone'].mean()
del india_uv['no']

australia_uv['no'] = np.arange(len(australia_uv))
australia_uv['local slope_uv'] = np.arange(len(australia_uv))
for i in np.arange(len(australia_uv)):
    slope=tangent(australia_uv['no'], australia_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    australia_uv['local slope_uv'][i] = slope
australia_uv['global slope_uv']=tangent(australia_uv['no'], australia_uv['UVIEF'],i,glob=True)
australia_uv['average_uv_index']=australia_uv['UVIEF'].mean()
australia_uv['average_ozone']=australia_uv['ozone'].mean()
del australia_uv['no']



tianjin_uv=df9[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
hebei_uv=df10[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
qinghai_uv=df11[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]
liaoning_uv=df12[['UVIEF','UVDEF','UVDEC','UVDDF','ozone']]

tianjin_uv['no'] = np.arange(len(tianjin_uv))
tianjin_uv['local slope_uv'] = np.arange(len(tianjin_uv))
for i in np.arange(len(tianjin_uv)):
    slope=tangent(tianjin_uv['no'], tianjin_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    tianjin_uv['local slope_uv'][i] = slope
tianjin_uv['global slope_uv']=tangent(tianjin_uv['no'], tianjin_uv['UVIEF'],i,glob=True)
tianjin_uv['average_uv_index']=tianjin_uv['UVIEF'].mean()
tianjin_uv['average_ozone']=tianjin_uv['ozone'].mean()
del tianjin_uv['no']


hebei_uv['no'] = np.arange(len(hebei_uv))
hebei_uv['local slope_uv'] = np.arange(len(hebei_uv))
for i in np.arange(len(hebei_uv)):
    slope=tangent(hebei_uv['no'], hebei_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    tianjin_uv['local slope_uv'][i] = slope
hebei_uv['global slope_uv']=tangent(hebei_uv['no'], hebei_uv['UVIEF'],i,glob=True)
hebei_uv['average_uv_index']=hebei_uv['UVIEF'].mean()
hebei_uv['average_ozone']=hebei_uv['ozone'].mean()
del hebei_uv['no']

qinghai_uv['no'] = np.arange(len(qinghai_uv))
qinghai_uv['local slope_uv'] = np.arange(len(qinghai_uv))
for i in np.arange(len(qinghai_uv)):
    slope=tangent(qinghai_uv['no'], qinghai_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    qinghai_uv['local slope_uv'][i] = slope
qinghai_uv['global slope_uv']=tangent(qinghai_uv['no'], qinghai_uv['UVIEF'],i,glob=True)
qinghai_uv['average_uv_index']=qinghai_uv['UVIEF'].mean()
qinghai_uv['average_ozone']=qinghai_uv['ozone'].mean()
del qinghai_uv['no']

liaoning_uv['no'] = np.arange(len(liaoning_uv))
liaoning_uv['local slope_uv'] = np.arange(len(liaoning_uv))
for i in np.arange(len(liaoning_uv)):
    liaoning=tangent(liaoning_uv['no'], liaoning_uv['UVIEF'],i)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(us['no'][0:i+1], us['confirmed'][0:i+1])  
    liaoning_uv['local slope_uv'][i] = slope
liaoning_uv['global slope_uv']=tangent(liaoning_uv['no'], liaoning_uv['UVIEF'],i,glob=True)
liaoning_uv['average_uv_index']=liaoning_uv['UVIEF'].mean()
liaoning_uv['average_ozone']=liaoning_uv['ozone'].mean()
del liaoning_uv['no']

frames_uv = [us_uv, it_uv, thai_uv,japan_uv, india_uv, australia_uv]
result_uv = pd.concat(frames_uv).fillna(0)
result_final = pd.concat([result,result_uv],axis=1).fillna(0)

frames_china_uv = [tianjin_uv, hebei_uv, qinghai_uv,liaoning_uv]
result_china_uv = pd.concat(frames_china_uv).fillna(0)
result_final_china = pd.concat([result_china,result_china_uv],axis=1).fillna(0)
#result_final = result_final[['confirmed', 'UVIEF','target']]


#sns.lmplot(x='confirmed', y='UVIEF', fit_reg=False, data=result_final, hue='target');
sum_corr = abs(result_final.corr()).sum().sort_values(ascending=True).index.values
corr=result_final[sum_corr].corr()

cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)
plt.figure(figsize=(15, 8))
corr=sns.heatmap(corr, annot=True, cmap=cmap);
corr.set_xticklabels(corr.get_xticklabels(), rotation=50, horizontalalignment='right')
plt.savefig('data/corr.png')


sum_corr_china = abs(result_final_china.corr()).sum().sort_values(ascending=True).index.values
corr_china=result_final[sum_corr_china].corr()

cmap = sns.diverging_palette(240, 10, n=9) 
plt.figure(figsize=(15, 8))
corr=sns.heatmap(corr_china, annot=True, cmap=cmap);
corr.set_xticklabels(corr.get_xticklabels(), rotation=50, horizontalalignment='right')
plt.savefig('data/corr_china.png')

def draw_tangent(x,y,a,glob=False):
 # interpolate the data with a spline
 spl = interpolate.splrep(x,y)
 if glob==False:
     small_t = np.arange(a-3,a+3)
 else:
     small_t = np.arange(a-30,a+3)
 fa = interpolate.splev(a,spl,der=0)     # f(a)
 fprime = interpolate.splev(a,spl,der=1) # f'(a)
 #print fprime
 tan = fa+fprime*(small_t-a) # tangent
 if glob==False:
     if i==0:
         plt.plot(a,fa,'om',small_t,tan,'--r',label='local slope')
     else:
         plt.plot(a,fa,'om',small_t,tan,'--r')
 else:
     plt.plot(a,fa,'bs',small_t,tan,'--b',label='global slope')
 
from scipy import interpolate
 
ax_ex=plt.figure(figsize=(15,10))
plt.plot(indo['no'],indo['confirmed'],label='_nolegend_')
indo.set_index('no')
for i in np.arange(len(indo['confirmed'])):
    draw_tangent(indo['no'], indo['confirmed'],i)
draw_tangent(indo['no'], indo['confirmed'],i,glob=True)
plt.legend()
#plt.xlabel('Day', fontsize=20);
#plt.ylabel('Confirmed', fontsize=20);
#plt.savefig('data/slopes.png')




'''
price_index = [21.0,17.6,19.3,28.9,21.1,20.5,22.1,26.4,22.3,24.4,24.6,28.0,24.7,24.9,25.7,31.6,39.1,31.3,31.3,32.1,34.4,38.0,36.7,39.6,58.8,71.8,57.7,62.6,63.8,66.3,63.6,81.0,109.5,92.7,91.3,116.0,101.5,96.1,116.0,119.1,153.5,162.6,144.6,141.5,154.6,174.3,174.7,180.6,174.1,185.2,193.1,196.3,202.3,238.6,228.1,231.1,247.7,273.1]
t = np.arange(1949,2007)
ax7=plt.figure(figsize=(15,10))
draw_tangent(t,price_index,1949)
draw_tangent(t,price_index,1950)
plt.plot(t,price_index,alpha=0.5)
'''
