# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:03:04 2020

@author: yudis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from numpy.random import rand
#matplotlib inline
import csv

df = pd.read_csv('data/waqi-covid19-airqualitydata_1.csv', skiprows=1,error_bad_lines=False)
df.head()
df.info()
df.columns = ['Date','Country','City','Specie','count','min','max','median','variance']
df.head
#print df.day
df.fillna
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)
######################################################################################
dfc = pd.read_csv('data/us-states.csv', skiprows=1)
dfc.head()
dfc.info()

dfc.columns = ['day','state', 'fips', 'confirmed','deaths']
dfc.head
#print df.day
dfc.fillna
dfc.day = pd.to_datetime(dfc.day)
dfc.set_index('day', inplace=True)

#######################################################################################
dfcc = pd.read_csv('data/time-series-19-covid-combined-2.csv', skiprows=1)
dfcc.head()
dfcc.info()

dfcc.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']
dfcc.head

dfcc.fillna
dfcc.day = pd.to_datetime(dfcc.day)
dfcc.set_index('day', inplace=True)
######################################################################################



#######################################################################################

is_jkt =  (df['City']=='Jakarta') & (df['Specie']=='pm25')
is_mln =  (df['City']=='Milan') & (df['Specie']=='pm25')
is_whn =  (df['City']=='Wuhan') & (df['Specie']=='pm25')
is_bjg =  (df['City']=='Beijing') & (df['Specie']=='pm25')
is_itb =  (df['City']=='Istanbul') & (df['Specie']=='pm25')
is_bkk =  (df['City']=='Bangkok') & (df['Specie']=='pm25')
is_adld =  (df['City']=='Adelaide') & (df['Specie']=='pm25')
is_london =  (df['City']=='Amsterdam') & (df['Specie']=='pm25')
is_paris =  (df['City']=='Paris') & (df['Specie']=='pm10')
is_oklm =  (df['City']=='Oklahoma City') & (df['Specie']=='pm10')
is_ral =  (df['City']=='Raleigh') & (df['Specie']=='pm10')
is_mem =  (df['City']=='Memphis') & (df['Specie']=='pm10')
is_jack =  (df['City']=='Jackson') & (df['Specie']=='pm10')
is_bos =  (df['City']=='Boston') & (df['Specie']=='pm10')
is_atl =  (df['City']=='Atlanta') & (df['Specie']=='pm10')
is_det =  (df['City']=='Detroit') & (df['Specie']=='pm25')
is_la =  (df['City']=='Los Angeles') & (df['Specie']=='pm25')
is_chi =  (df['City']=='Chicago') & (df['Specie']=='pm25')
is_ber =  (df['City']=='Berlin') & (df['Specie']=='pm25')
is_bru =  (df['City']=='Brussels') & (df['Specie']=='pm25')
is_tyo =  (df['City']=='Tokyo') & (df['Specie']=='pm25')
is_mad =  (df['City']=='Madrid') & (df['Specie']=='pm25')
is_seo =  (df['City']=='Seoul') & (df['Specie']=='pm25')

is_kl =  (df['City']=='Chiang Mai') & (df['Specie']=='pm25')
is_sin =  (df['City']=='Singapore') & (df['Specie']=='pm25')
is_nd =  (df['City']=='New Delhi') & (df['Specie']=='pm25')
is_mum =  (df['City']=='Mumbai') & (df['Specie']=='pm25')
is_col =  (df['City']=='Colombo') & (df['Specie']=='pm25')
is_man =  (df['City']=='Sydney') & (df['Specie']=='pm25')
is_ho =  (df['City']=='Ho Chi Minh City') & (df['Specie']=='pm25')
is_han =  (df['City']=='Hanoi') & (df['Specie']=='pm25')
is_auck =  (df['City']=='Auckland') & (df['Specie']=='pm25')
is_quit =  (df['City']=='Quito') & (df['Specie']=='pm25')
is_mex =  (df['City']=='Mexico City') & (df['Specie']=='pm25')

dfjkt= df[(is_jkt)][['median']]
dfjkt=dfjkt.sort_values('Date')
dfitb= df[(is_itb)][['median']]
dfitb=dfitb.sort_values('Date')
dfwhn= df[(is_whn)][['median']]
dfwhn=dfwhn.sort_values('Date')
dfbjg= df[(is_bjg)][['median']]
dfbjg=dfbjg.sort_values('Date')
dfbkk= df[(is_bkk)][['median']]
dfbkk=dfbkk.sort_values('Date')
dfadld= df[(is_adld)][['median']]
dfadld=dfadld.sort_values('Date')
dfmln= df[(is_mln)][['median']]
dfmln=dfmln.sort_values('Date')
dfparis= df[(is_paris)][['median']]
dfparis=dfparis.sort_values('Date')
dflondon= df[(is_london)][['median']]
dflondon=dflondon.sort_values('Date')
dfkl= df[(is_kl)][['median']]
dfkl=dfkl.sort_values('Date')
dfnd= df[(is_nd)][['median']]
dfnd=dfnd.sort_values('Date')
dfmum= df[(is_mum)][['median']]
dfmum=dfmum.sort_values('Date')
dfcol= df[(is_col)][['median']]
dfcol=dfcol.sort_values('Date')
dfman= df[(is_man)][['median']]
dfman=dfman.sort_values('Date')
dfsin= df[(is_sin)][['median']]
dfsin=dfsin.sort_values('Date')
dfho= df[(is_ho)][['median']]
dfho=dfho.sort_values('Date')
dfhan= df[(is_han)][['median']]
dfhan=dfhan.sort_values('Date')
dfauck= df[(is_auck)][['median']]
dfauck=dfauck.sort_values('Date')
dfber= df[(is_ber)][['median']]
dfber=dfber.sort_values('Date')
dfbru= df[(is_bru)][['median']]
dfbru=dfbru.sort_values('Date')
dftyo= df[(is_tyo)][['median']]
dftyo=dftyo.sort_values('Date')
dfquit= df[(is_quit)][['median']]
dfquit=dfquit.sort_values('Date')
dfmex= df[(is_mex)][['median']]
dfmex=dfmex.sort_values('Date')
dfseo= df[(is_seo)][['median']]
dfseo=dfseo.sort_values('Date')





dfoklm= df[(is_oklm)][['median']]
dfoklm=dfoklm.sort_values('Date')
dfral= df[(is_ral)][['median']]
dfral=dfral.sort_values('Date')
dfmem= df[(is_mem)][['median']]
dfmem=dfmem.sort_values('Date')
dfjack= df[(is_jack)][['median']]
dfjack=dfjack.sort_values('Date')
dfbos= df[(is_bos)][['median']]
dfbos=dfbos.sort_values('Date')
dfatl= df[(is_atl)][['median']]
dfatl=dfatl.sort_values('Date')
dfdet= df[(is_det)][['median']]
dfdet=dfdet.sort_values('Date')
dfla= df[(is_la)][['median']]
dfla=dfla.sort_values('Date')
dfchi= df[(is_chi)][['median']]
dfchi=dfchi.sort_values('Date')
dfmad= df[(is_mad)][['median']]
dfmad=dfmad.sort_values('Date')



mix=pd.concat([dfjkt,dfbkk,dfwhn,dfbjg,dfsin,dftyo,dfmln,dfbru,dfman,dfadld], axis=1)
mix.head()
mix.columns = ['Jakarta','Bangkok', 'Wuhan','Beijing','Singapore','Tokyo','Milan','Brussels','Sydney','Adelaide']

fig_st, ax_st = plt.subplots(figsize=(10,5))
sns.boxplot(data=mix,ax=ax_st)
#sns.boxplot(data=dfbkk)
ax_st.set_ylabel('pm2.5')
ax_st.set_title('The rate of pm2.5 density')

plt.savefig('data/boxplotpm25.png')
#################################################################################################
fig, ax0 = plt.subplots(figsize=(15,10))
#ax0.plot(dfjkt,color='blue', label='Jakarta')
#ax0.plot(dfmln,color='red', label='Milan')
ax0.plot(dfwhn,color='green', label='Wuhan')
ax0.plot(dfbjg,color='purple', label='Beijing')
#ax0.plot(dfmad,color='orange', label='Ho Chi Minh')
#ax0.plot(dfitb,color='brown', label='Istanbul')
#ax0.plot(dfbkk,color='orange', label='Bangkok')
#ax0.plot(dfadld,color='slateblue', label='Adelaide')
#ax0.plot(dflondon,color='grey', label='Amsterdam')
#ax0.plot(dfparis,color='yellow', label='Paris')
ax0.axvline(x='2020-01-23', color='k', linestyle='--', label='The day of Hubei Lockdown')
ax0.axvline(x='2020-02-10', color='blue', linestyle='--', label='Closed management of communities in Beijing')
ax0.axvline(x='2020-02-06', color='silver', linestyle='--', label='The day of Pollutant Degradation in Wuhan')
ax0.axvline(x='2020-02-13', color='r', linestyle='--', label='The day of Pollutant Degradation in Beijing')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('pm2.5', fontsize=20);
plt.savefig('data/pm25wuhanbeijing.png')
#######################################################################################


is_hub =  dfcc['territory']=='Hubei'
dfchub= dfcc[(is_hub)][['confirmed']]
dfchub=dfchub.sort_values('day')
is_bei =  dfcc['territory']=='Beijing'
dfcbei= dfcc[(is_bei)][['confirmed']]
dfcbei=dfcbei.sort_values('day')
is_vie =  dfcc['country']=='Vietnam'
dfcvie= dfcc[(is_vie)][['confirmed']]
dfcvie=dfcvie.sort_values('day')

fig, axk = plt.subplots(figsize=(15,10))
axk.plot(dfchub,color='green', label='Hubei')
axk.axvline(x='2020-02-06', color='silver', linestyle='--', label='The day of Pollutant Degradation in Wuhan')

#axk.plot(dfcbei,color='purple', label='California')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('confirmed', fontsize=20);
plt.savefig('data/confirmedhubei.png')

#######################################################################################
fig, axl = plt.subplots(figsize=(15,10))
axl.plot(dfcbei,color='purple', label='Beijing')
axl.axvline(x='2020-02-13', color='r', linestyle='--', label='The day of Pollutant Degradation in Beijing')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('confirmed', fontsize=20);

plt.savefig('data/confirmedbeijing.png')
#######################################################################################

fig, axm = plt.subplots(figsize=(15,10))
axm.plot(dfcvie,color='orange', label='Vietnam')
axm.axvline(x='2020-02-28', color='r', linestyle='--', label='The day of Pollutant Degradation in Ho Chi Minh')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('confirmed', fontsize=20);

plt.savefig('data/confirmedvietnam.png')

###############################################################################



fig, ax1 = plt.subplots(figsize=(15,10))
is_oklm =  dfc['state']=='Oklahoma'
dfcoklm= dfc[(is_oklm)][['confirmed']]
dfcoklm=dfcoklm.sort_values('day')
is_ral =  dfc['state']=='North Carolina'
dfcral= dfc[(is_ral)][['confirmed']]
dfcral=dfcral.sort_values('day')
is_ten =  dfc['state']=='Tennessee'
dfcten= dfc[(is_ten)][['confirmed']]
dfcten=dfcten.sort_values('day')
is_mis =  dfc['state']=='Mississippi'
dfcmis= dfc[(is_mis)][['confirmed']]
dfcmis=dfcmis.sort_values('day')
is_nyc =  dfc['state']=='Massachusetts'
dfcnyc= dfc[(is_nyc)][['confirmed']]
dfcnyc=dfcnyc.sort_values('day')
is_geor =  dfc['state']=='Georgia'
dfcgeor= dfc[(is_geor)][['confirmed']]
dfcgeor=dfcgeor.sort_values('day')
is_mic =  dfc['state']=='Michigan'
dfcmic= dfc[(is_mic)][['confirmed']]
dfcmic=dfcmic.sort_values('day')
is_cal =  dfc['state']=='California'
dfccal= dfc[(is_cal)][['confirmed']]
dfccal=dfccal.sort_values('day')


ax1.plot(dfcoklm,color='blue', label='Oklahoma')
ax1.plot(dfcral,color='green', label='North Carolina')
ax1.plot(dfcten,color='red', label='Tennesee')
ax1.plot(dfcmis,color='cyan', label='Mississippi')
ax1.plot(dfcnyc,color='magenta', label='Massachusetts')
ax1.plot(dfcgeor,color='yellow', label='Georgia')
ax1.plot(dfcmic,color='black', label='Michigan')
ax1.plot(dfccal,color='grey', label='California')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('confirmed', fontsize=20);
plt.savefig('data/covid_usa.png')
#######################################################################################

fig, ax3 = plt.subplots(figsize=(15,10))
ax3.plot(dfjkt,color='yellow', label='Jakarta')
ax3.plot(dfbkk,color='blue', label='Bangkok')
ax3.plot(dfsin,color='brown', label='Singapore')
ax3.plot(dfkl,color='green', label='Chiang Mai')
ax3.plot(dfnd,color='red', label='New Delhi')
ax3.plot(dfman,color='slateblue', label='Sydney')
ax3.plot(dfmum,color='orange', label='Mumbai')
ax3.plot(dfcol,color='grey', label='Colombo')
ax3.plot(dfadld,color='purple', label='Adelaide')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('pm2.5', fontsize=20);
plt.savefig('data/pm25_tropis.png')
######################################################################################

fig, ax4 = plt.subplots(figsize=(15,10))
ax4.plot(dfoklm,color='blue', label='Oklahoma (Oklahoma)')
ax4.plot(dfral,color='green', label='Raleigh (North Carolina)')
#ax4.plot(dfmem,color='red', label='Memphis (Tennesee)')
#ax4.plot(dfjack,color='cyan', label='Jackson (Mississippi)')
#ax4.plot(dfbos,color='magenta', label='Boston (Massachusetts)')
ax4.plot(dfatl,color='yellow', label='Atlanta (Georgia)')
ax4.plot(dfdet,color='black', label='Detroit (Michigan)')
ax4.plot(dfla,color='grey', label='Los Angeles (California)')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('pm10', fontsize=20);
plt.savefig('data/pm210_usa.png')

#####################################################################################

data = ['Jakarta', 'Bangkok', 'Singapore', 'New Delhi', 'Sydney', 'Mumbai', 'Ho Chi Minh', 'Hanoi','Auckland','Adelaide','Quito','Mexico City']
confirm = [2044.0, 1262.0, 2299.0, 1640, 222.0, 1008.0, 54.0, 124.0, 176.0, 336.0, 819.0, 2299.0]
count=[1,2,3,4,5,6,7,8,9,10,11,12]
data2 = ['Los Angeles','Detroit','Chicago','Berlin','Brussels','Amsterdam','Tokyo','Madrid','Milan','Seoul','Wuhan']
confirm2 = [9420.0, 6781.0, 9113.0,4028.0,221.0,500.0, 2319.0,49000.0,66236.0,10000,55000]




fig, ax = plt.subplots(figsize=(15,10))
 
# Get a color map
my_cmap = cm.get_cmap('jet')

 
# Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
my_norm = Normalize(vmin=0, vmax=12)
 
ax.bar(data, confirm, color=my_cmap(my_norm(count)))

fig.savefig('data/barchart_tropis.png')
#######################################################################################
no=np.arange(8)
reg =pd.DataFrame({ 'data':data,  'confirm':confirm})
reg['pm25']=[dfjkt.mean()[0],dfbkk.mean()[0],dfsin.mean()[0],dfnd.mean()[0],dfman.mean()[0],dfmum.mean()[0],dfho.mean()[0],dfhan.mean()[0], dfauck.mean()[0], dfadld.mean()[0], dfquit.mean()[0],dfmex.mean()[0]]
reg['region']='tropical'
reg2 =pd.DataFrame({ 'data':data2,  'confirm':confirm2})
reg2['pm25']=[dfla.mean()[0],dfdet.mean()[0],dfchi.mean()[0],dfber.mean()[0],dfbru.mean()[0],dflondon.mean()[0],dftyo.mean()[0],dfmad.mean()[0],dfmln.mean()[0],dfseo.mean()[0],dfwhn.mean()[0]]
reg2['region']='northern subtropical'

frames = [reg, reg2]

fin = pd.concat(frames)





#reg.set_index('no',inplace=True)
sns.set()

# Load the iris dataset
#iris = sns.load_dataset("iris")

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(x="pm25", y="confirm", hue="region",
               height=10, data=reg)
gax=g.axes[0,0]

for line in range(0,reg.shape[0]):
     gax.text(reg.pm25[line]+0.2, reg.confirm[line], reg.data[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
#gax.set_ylim([0.0,0.0006])
# Use more informative axis labels than are provided by default
g.set_axis_labels("Average PM2.5", "Total confirmed per area size (April 12, 2020)").set(ylim=(0.0))


g.savefig('data/regres_tropis.png') 

g2 = sns.lmplot(x="pm25", y="confirm", hue="region",
               height=10, data=reg2)

gax2=g2.axes[0,0]

for line in range(0,reg2.shape[0]):
     gax2.text(reg2.pm25[line]+0.2, reg2.confirm[line], reg2.data[line], horizontalalignment='left', size='medium', color='black', weight='semibold')


# Use more informative axis labels than are provided by default
g2.set_axis_labels("Average PM2.5", "Total confirmed per area size (April 12, 2020)").set(ylim=(0.0))


g2.savefig('data/regres_subtropis.png') 