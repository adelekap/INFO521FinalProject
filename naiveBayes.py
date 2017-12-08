from sklearn import datasets
import data
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from SVM import plot_contours, make_meshgrid
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB


def plot_nb_results(model,xx,yy,cv,title):
    fig = plt.figure()
    fig.figsize = (14, 10)
    ax = plt.gca()
    colors = ['#020202' if rat == 6 else '#676767' if rat == 15 else '#f9f9f9' for rat in list(targets)]
    plot_contours(ax, model, xx, yy,
                  cmap=plt.get_cmap('Accent'), alpha=0.8)
    ax.scatter(d['Water Maze CIPL'], d['Working Memory CIPL'], c=colors)
    cv = np.mean(cross_val_score(model, X_test, y_test, cv=10))
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_ylabel('Spatial Memory')
    ax.set_xlabel('Working Memory')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title+' (CV =' + str(cv.round(2)) + ')', fontsize=12)

    plt.tight_layout()
    plt.savefig('Results/Classification/NaiveBayes/NaiveBayes.pdf')
    plt.show()


if __name__ == '__main__':
    d = data.allData[['Water Maze CIPL','Working Memory CIPL']]
    targets = data.allData['Age']
    X_train,X_test,y_train,y_test = train_test_split(d,targets,random_state=1)
    xx, yy = make_meshgrid(d['Water Maze CIPL'], d['Working Memory CIPL'])

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gcv = np.mean(cross_val_score(gnb,X_test,y_test,cv=10))
    plot_nb_results(gnb,xx,yy,gcv,'Naive Bayes')