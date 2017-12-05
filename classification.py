from sklearn.neighbors import KNeighborsClassifier
import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_classifications(data,responses,title=None):
    colors = ['#020202' if rat == 6 else '#676767' if rat == 15 else '#f9f9f9' for rat in list(responses)]
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(data['Working Memory CIPL'], data['Water Maze CIPL'], c=colors)
    plt.xlabel('Working Memory CIPL')
    blDot = mlines.Line2D([],[],color='#020202',marker='o',label='Young',linestyle=' ')
    grDot = mlines.Line2D([],[],color='#676767',marker='o',label='Middle',linestyle = ' ')
    whDot = mlines.Line2D([], [], color='#f9f9f9', marker='o', label='Old',linestyle= ' ')
    plt.legend(handles=[blDot,grDot,whDot],numpoints=1,loc=4)
    plt.title(title)
    plt.show()

# Divide data into training set (75%) and testing set (25%)
msk = np.random.rand(len(data.allData)) < 0.75
trainSet = data.allData[msk]
testSet = data.allData[~msk]

plot_classifications(trainSet[['Working Memory CIPL','Water Maze CIPL']],trainSet['Age'],'Training Set')

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(trainSet[['Working Memory CIPL','Water Maze CIPL']],trainSet['Age'])
predictions = knn.predict(testSet[['Working Memory CIPL','Water Maze CIPL']])

actual = testSet['Age']
print('debug')
