from sklearn import svm
from sklearn.model_selection import train_test_split
import data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary to plot in 3D
from sklearn.model_selection import cross_val_score


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


if __name__ == '__main__':
    features = ['Working Memory CIPL', 'Water Maze CIPL']
    X = data.allData[features]
    y = data.allData['Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=2, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 2) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    fig.figsize = (14,10)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X_train[features[0]], X_train[features[1]]
    xx, yy = make_meshgrid(X0, X1)


    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.get_cmap('Accent'), alpha=0.8)
        colors = ['#020202' if rat == 6 else '#676767' if rat == 15 else '#f9f9f9' for rat in list(y)]
        ax.scatter(X['Working Memory CIPL'],X['Water Maze CIPL'],c=colors)
        cv = np.mean(cross_val_score(clf,X_test,y_test,cv=10))
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_ylabel('Spatial Memory')
        ax.set_xlabel('Working Memory')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title +' (CV ='+str(cv.round(2))+')',fontsize=12)

    plt.tight_layout()
    plt.savefig('Results/Classification/SVM/SVMs.pdf')
    plt.show()

