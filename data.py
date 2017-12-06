import pandas as pd

"""
This module extracts and cleans the data from the csvs.
"""

dataDir = 'Data/'

# ||| WORKING MEMORY DATA |||
wmData = pd.read_csv(dataDir+'CAS_WorkingMemory.csv')[['Rat ID','Age','Trial','Platform : CIPL']]
wmData.columns = ['Rat ID','Age','Trial','Working Memory CIPL']
wmData = wmData[wmData['Trial'].isin(range(2,20,2))] # We only want the retention trial data
# Trials 2,4,and 6 are 30 second delay
# Trials 8,12,and 16 are 30 minute delay
# Trials 10,14,and 18 are 2 hour delay
column_name = 'Trial'
wmData.loc[wmData.Trial.isin([2,4,6]), column_name] = '30 second'
wmData.loc[wmData.Trial.isin([8,12,16]), column_name] = '30 minute'
wmData.loc[wmData.Trial.isin([10,14,18]), column_name] = '2 hour'

#Get list of rats that completed working memory task
rats = list(wmData['Rat ID'].unique())


# ||| WATER MAZE DATA |||
wmazeDataAll = pd.read_csv(dataDir+'CAS_Watermaze.csv')[['Rat ID','Age','Trial','CIPL (m*sec)','TrialType']]
# We only want the spatial test data
wmazeData = (wmazeDataAll[wmazeDataAll['TrialType'] == 'Spatial'])[['Rat ID','Age','Trial','CIPL (m*sec)']]
wmazeData.columns = ['Rat ID','Age','Trial','Water Maze CIPL']
wmazeData = wmazeData[wmazeData['Rat ID'].isin(rats)]  # Only get the rats that also participated in working memory

# ||| COMBINED DATA |||
avgs = wmData[['Rat ID','Working Memory CIPL']].groupby('Rat ID').mean()
allData = pd.DataFrame()
allData['Working Memory CIPL'] = avgs['Working Memory CIPL']
allData['Rat ID'] = avgs.index.values

watermazeAvgs = wmazeData.groupby('Rat ID').mean()
temp = pd.DataFrame()
temp['Water Maze CIPL'] = watermazeAvgs['Water Maze CIPL']
temp['Rat ID'] = temp.index.values
key = pd.DataFrame()
key['Age'] = watermazeAvgs['Age']
key['Rat ID'] = watermazeAvgs.index.values
allData = pd.merge(allData,temp,on='Rat ID')
allData = pd.merge(allData,key,on='Rat ID')

#Working Memory Thirty Second Delay
thirtySec = wmData[wmData['Trial'] == '30 second'][['Rat ID','Age','Working Memory CIPL']].groupby('Rat ID').mean()
thirtySec['Rat ID'] = thirtySec.index.values
thirtySec = pd.merge(temp,thirtySec,on='Rat ID')

#Working Memory Thirty Minute Delay
thirtyMin = wmData[wmData['Trial'] == '30 minute'][['Rat ID','Age','Working Memory CIPL']].groupby('Rat ID').mean()
thirtyMin['Rat ID'] = thirtyMin.index.values
thirtyMin = pd.merge(temp,thirtyMin,on='Rat ID')

#Working Memory Two Hour Delay
twoHr = wmData[wmData['Trial'] == '2 hour'][['Rat ID','Age','Working Memory CIPL']].groupby('Rat ID').mean()
twoHr['Rat ID'] = twoHr.index.values
twoHr = pd.merge(temp,twoHr,on='Rat ID')






