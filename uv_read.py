# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:40:30 2020

@author: yudis
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import csv

# read flash.dat to a list of lists
'''
datContent = [i.strip().split() for i in open("./data/uv_Buenos_Aires_Argentina.dat").readlines()]

# write it as a new CSV file
with open("./data/uv_Buenos_Aires_Argentina.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(datContent)


'''
df1 = pd.read_csv('data/uv_AcadiaNatForest_USA_1.csv', skiprows=1)
df1.head()
df1.info()

df2 = pd.read_csv('data/uv_Adelaide_Australia_1.csv', skiprows=1)
df2.head()
df2.info()

df3 = pd.read_csv('data/uv_Atlanta_USA_1.csv', skiprows=1)
df3.head()
df3.info()

df4 = pd.read_csv('data/uv_Chiang_Mai_Thailand_1.csv', skiprows=1)
df4.head()
df4.info()

df5 = pd.read_csv('data/uv_Gibilmanna_Italy_1.csv', skiprows=1)
df5.head()
df5.info()

df6 = pd.read_csv('data/uv_Naha_Japan_1.csv', skiprows=1)
df6.head()
df6.info()

df7 = pd.read_csv('data/uv_Hyderabad_India_1.csv', skiprows=1)
df7.head()
df7.info()

df8 = pd.read_csv('data/uv_Obninsk_Russia_1.csv', skiprows=1)
df8.head()
df8.info()

df9 = pd.read_csv('data/uv_Tianjin_China_1.csv', skiprows=1)
df9.head()
df9.info()

df10 = pd.read_csv('data/uv_Baoding_China_1.csv', skiprows=1)
df10.head()
df10.info()

df11 = pd.read_csv('data/uv_MountWaliguan_China_1.csv', skiprows=1)
df11.head()
df11.info()

df12 = pd.read_csv('data/uv_Dalian_China_1.csv', skiprows=1)
df12.head()
df12.info()

df12 = pd.read_csv('data/uv_Dalian_China_1.csv', skiprows=1)
df12.head()
df12.info()

df13 = pd.read_csv('data/uv_Mecca_Saudi_Arabia_1.csv', skiprows=1)
df13.head()
df13.info()

df14 = pd.read_csv('data/uv_Durban_SouthAfrica_1.csv', skiprows=1)
df14.head()
df14.info()


df15 = pd.read_csv('data/uv_Buenos_Aires_Argentina_1.csv', skiprows=1)
df15.head()
df15.info()


df1.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df1.head
#print df.day
df1.YYYYMMDD = pd.to_datetime(df1.YYYYMMDD,format='%Y%m%d', errors='ignore')
df1.set_index('YYYYMMDD', inplace=True)
print df1
df1= df1.loc['20200122':]
#ax=plt.figure(figsize=(15,10))
#plt.plot(df[['UVIEF']], label='USA')
#plt.legend(loc=2)
#plt.xlabel('Day', fontsize=20);
#plt.ylabel('UV Index', fontsize=20);

df2.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df2.head
#print df.day
df2.YYYYMMDD = pd.to_datetime(df2.YYYYMMDD,format='%Y%m%d', errors='ignore')
df2.set_index('YYYYMMDD', inplace=True)
print df2
df2= df2.loc['20200122':]


df3.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df3.head
#print df.day
df3.YYYYMMDD = pd.to_datetime(df3.YYYYMMDD,format='%Y%m%d', errors='ignore')
df3.set_index('YYYYMMDD', inplace=True)
print df3
df3= df3.loc['20200122':]

df4.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df4.head

df4.YYYYMMDD = pd.to_datetime(df4.YYYYMMDD,format='%Y%m%d', errors='ignore')
df4.set_index('YYYYMMDD', inplace=True)
print df4
df4= df4.loc['20200122':]

df5.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df5.head

df5.YYYYMMDD = pd.to_datetime(df5.YYYYMMDD,format='%Y%m%d', errors='ignore')
df5.set_index('YYYYMMDD', inplace=True)
print df5
df5= df5.loc['20200122':]

df6.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df6.head

df6.YYYYMMDD = pd.to_datetime(df6.YYYYMMDD,format='%Y%m%d', errors='ignore')
df6.set_index('YYYYMMDD', inplace=True)
print df6
df6= df6.loc['20200122':]

df7.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df7.head
#print df.day
df7.YYYYMMDD = pd.to_datetime(df7.YYYYMMDD,format='%Y%m%d', errors='ignore')
df7.set_index('YYYYMMDD', inplace=True)
print df7
df7= df7.loc['20200122':]

df8.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df8.head
#print df.day
df8.YYYYMMDD = pd.to_datetime(df8.YYYYMMDD,format='%Y%m%d', errors='ignore')
df8.set_index('YYYYMMDD', inplace=True)
print df8
df8= df8.loc['20200122':]

df9.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df9.head
df9.YYYYMMDD = pd.to_datetime(df9.YYYYMMDD,format='%Y%m%d', errors='ignore')
df9.set_index('YYYYMMDD', inplace=True)
print df9
df9= df9.loc['20200122':'20200328']

df10.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df10.head
df10.YYYYMMDD = pd.to_datetime(df10.YYYYMMDD,format='%Y%m%d', errors='ignore')
df10.set_index('YYYYMMDD', inplace=True)
print df10
df10= df10.loc['20200122':'20200328']

df11.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df11.head
df11.YYYYMMDD = pd.to_datetime(df11.YYYYMMDD,format='%Y%m%d', errors='ignore')
df11.set_index('YYYYMMDD', inplace=True)
print df11
df11= df11.loc['20200122':'20200328']

df12.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df12.head
df12.YYYYMMDD = pd.to_datetime(df12.YYYYMMDD,format='%Y%m%d', errors='ignore')
df12.set_index('YYYYMMDD', inplace=True)
print df12
df12= df12.loc['20200122':'20200328']

df13.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df13.head
df13.YYYYMMDD = pd.to_datetime(df13.YYYYMMDD,format='%Y%m%d', errors='ignore')
df13.set_index('YYYYMMDD', inplace=True)
print df13
df13= df13.loc['20200122':'20200328']

df14.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df14.head
df14.YYYYMMDD = pd.to_datetime(df14.YYYYMMDD,format='%Y%m%d', errors='ignore')
df14.set_index('YYYYMMDD', inplace=True)
print df14
df14= df14.loc['20200122':'20200328']

df15.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
df15.head
df15.YYYYMMDD = pd.to_datetime(df15.YYYYMMDD,format='%Y%m%d', errors='ignore')
df15.set_index('YYYYMMDD', inplace=True)
print df15
df15= df15.loc['20200122':'20200328']


ax=plt.figure(figsize=(15,10))
plt.plot(df1[['UVIEF']], label='USA-New York')
plt.plot(df3[['UVIEF']], label='USA-Atlanta')
plt.plot(df2[['UVIEF']], label='Australia')
plt.plot(df4[['UVIEF']], label='Thailand')
plt.plot(df5[['UVIEF']], label='Italy')
plt.plot(df6[['UVIEF']], label='Japan')
plt.plot(df7[['UVIEF']], label='India')
plt.plot(df8[['UVIEF']], label='Russia-Obninsk')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('UV Index', fontsize=20);
#plt.savefig('data/uv_index.png')


ax2=plt.figure(figsize=(15,10))
plt.plot(df9[['UVIEF']], label='Tianjin')
plt.plot(df10[['UVIEF']], label='Baoding')
plt.plot(df11[['UVIEF']], label='Mount Waliguan')
plt.plot(df12[['UVIEF']], label='Dalian')


plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('UV Index', fontsize=20);
#plt.savefig('data/china_uv_index.png')

ax3=plt.figure(figsize=(15,10))
plt.plot(df1[['UVIEF']], color='blue',label='New York-USA')
plt.plot(df3[['UVIEF']], color='blue',label='Atlanta-USA')
plt.plot(df5[['UVIEF']], color='blue',label='Gybilmanna-Italy')
plt.plot(df6[['UVIEF']], color='blue',label='Naha-Japan')
plt.plot(df8[['UVIEF']], color='blue',label='Obninsk-Russia')
plt.plot(df10[['UVIEF']], color='blue',label='Baoding-China')
plt.plot(df4[['UVIEF']], color='green',label='Chiang Mai-Thailand')
plt.plot(df7[['UVIEF']], color='green',label='Hyderabad-India')
plt.plot(df13[['UVIEF']], color='green',label='Mecca-Saudi Arabia')
plt.plot(df14[['UVIEF']], color='red',label='Durban-South Africa')
plt.plot(df2[['UVIEF']], color='red',label='Adelaide-Australia')
plt.plot(df15[['UVIEF']], color='red',label='Buenos Aires-Argentina')

plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('UV Index', fontsize=20);

plt.savefig('data/uv_season.png')