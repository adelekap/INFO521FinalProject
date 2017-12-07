from sklearn.neighbors import KNeighborsClassifier
import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import metrics
from sklearn.model_selection import cross_val_score,KFold

"""
This module fits the data to a K-Nearest Neighbors model
to classify age using cognitive performance.
"""

def plot_training_and_testing(trainSet,testSet):
    """
    Plots the training set and the testing set.
    :param trainSet: training set data
    :param testSet: testing set data
    :return: None
    """
    plot_classifications(trainSet[['Working Memory CIPL','Water Maze CIPL']],trainSet['Age'],'TrainingSet')
    plot_classifications(testSet[['Working Memory CIPL','Water Maze CIPL']],testSet['Age'],'TestingSet')


def plot_classifications(data,responses,title='Plot',marker='o',col=None,accuracy=None):
    """
    Plots the classification outputs from the model
    :param data: data set
    :param responses: classified response
    :param title: title of plot
    :param marker: marker type (default is dot)
    :param col: sequence of coloring for marker outline
    :param accuracy: accuarcy measure to be reported on the graph
    :return: None
    """
    colors = ['#020202' if rat == 6 else '#676767' if rat == 15 else '#f9f9f9' for rat in list(responses)]
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    ax.scatter(data['Working Memory CIPL'], data['Water Maze CIPL'], c=colors,marker=marker,s=50,edgecolors=col)
    plt.xlabel('Working Memory CIPL')
    plt.ylabel('Spatial Memory CIPL')
    blDot = mlines.Line2D([],[],color='#020202',marker=marker,label='Young',linestyle=' ')
    grDot = mlines.Line2D([],[],color='#676767',marker=marker,label='Middle',linestyle=' ')
    whDot = mlines.Line2D([], [], color='#f9f9f9', marker=marker, label='Old',linestyle=' ')
    plt.legend(handles=[blDot,grDot,whDot],numpoints=1,loc=4)
    plt.title(title)
    if accuracy:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95,'CLASSIFICATON ACCURACY: '+accuracy, transform=ax.transAxes, fontsize=14,
                verticalalignment='top',bbox=props,horizontalalignment='left')
    plt.xlim(0,30)
    plt.ylim(0,45)
    plt.tight_layout()
    plt.savefig('Results/Classification/KNN/'+title+'.pdf')
    plt.show()


def find_best_K(trainSet,testSet):
    """
    Trains models with various Ks and determines the best value for K
    :param trainSet: training data set
    :param testSet: testing data set
    :return: best value for k and accuracies
    """
    accuracy = []
    for n in range(1, 201):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(trainSet[['Working Memory CIPL', 'Water Maze CIPL']], trainSet['Age'])
        pred = knn.predict(testSet[['Working Memory CIPL', 'Water Maze CIPL']])
        actual = list(testSet['Age'])
        accuracy.append(metrics.accuracy_score(actual, list(pred)))
    bestK = accuracy.index(max(accuracy))
    print('Optimal K: ' + str(bestK))
    plt.plot(range(1, 201), accuracy, linestyle='-')
    plt.xlabel('Value of K')
    plt.ylabel('Testing Accuracy')
    plt.savefig('Results/Classification/KNN/EvaluateKs.pdf')
    plt.show()
    return bestK,accuracy




if __name__ == '__main__':
    # Divide data into training set (80%) and testing set (20%)
    np.random.seed(0)
    msk = np.random.rand(len(data.allData)) < 0.80
    trainSet = data.allData[msk]
    testSet = data.allData[~msk]
    plot_training_and_testing(trainSet,testSet)

    # Train Model
    bestK,accuracies = find_best_K(trainSet,testSet)
    knn = KNeighborsClassifier(n_neighbors=bestK)
    knn.fit(trainSet[['Working Memory CIPL','Water Maze CIPL']],trainSet['Age'])
    predictions = knn.predict(testSet[['Working Memory CIPL','Water Maze CIPL']])
    actual = list(testSet['Age'])

    # Cross Validation
    scores = cross_val_score(knn, data.allData[['Working Memory CIPL', 'Water Maze CIPL']], data.allData['Age'], cv=5)
    kf = KFold(n_splits=100)
    split = kf.split(data.allData[['Working Memory CIPL','Water Maze CIPL']])

    cv = np.mean([knn.score(testSet[['Working Memory CIPL', 'Water Maze CIPL']],testSet['Age']) for train, test in
           kf.split(data.allData[['Working Memory CIPL', 'Water Maze CIPL']])])

    # Set edge colors to be red if incorrectly predicted and plot
    col = ['#020202' if predictions[n]==actual[n] else '#ce0c2c' for n in range(len(actual))]
    plot_classifications(testSet[['Working Memory CIPL','Water Maze CIPL']],predictions,'Model Predictions',
                         marker='^',col = col,accuracy=str(cv))


