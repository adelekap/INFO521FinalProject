import pandas as pd


if __name__ == '__main__':
    dir = '/Users/adelekap/Documents/BarnesLab/CAS/CAS_Analysis/'

    allData = pd.DataFrame()

    for cas in [1,2,3,4,5,6,7,8,9,11,12,13,14]:  # Data not available for cohort 10
        data = pd.read_csv('{0}CAS{1}_WM.csv'.format(dir,str(cas)))[['Test','Animal','Age','Trial','Duration','Distance',
                                                                     'Mean speed','Path efficiency','Platform : entries',
                                                                     'Platform : CIPL']]
        key = pd.read_csv('{0}CAS{1}_Key.csv'.format(dir,str(cas)))[['AnyMaze #','Rat No']]
        key.columns = ['Animal','Rat ID']
        combined = pd.merge(data,key, on ='Animal')
        allData = allData.append(combined)

    allData.to_csv('CAS_WorkingMemory.csv')