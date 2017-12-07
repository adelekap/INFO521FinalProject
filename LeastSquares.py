from sklearn.model_selection import train_test_split
import data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessary to plot in 3D


def formula(xs,intercept, coefficients):
    """
    Calculates y value for given x's and model parameters
    :param xs: array of x inputs
    :param intercept: y intercept parameter
    :param coefficients: list of coefficients
    :return: list of y's
    """
    ys = []
    for x in xs:
        y = intercept
        for coefficient in coefficients:
            y += x*coefficient
        ys.append(y)
    return ys


def plot_fit_3D(intercept, coefficients,x_range,x_train,y_train,title='regPlot'):
    """
    Plots a scatter plot of the training data and the line of regression for two features.
    :param intercept: y intercept parameter
    :param coefficients: array of coefficients [working memory, spatial memory]
    :param x_range: list of x's
    :param x_train: training data
    :param y_train: training responses
    :param title: plot title
    :return: None
    """
    xx, yy = np.meshgrid(x_range, x_range)
    z = formula(x_range,intercept,coefficients)

    fig = plt.figure(figsize=(14,10))
    ax = fig.gca(projection='3d')
    ax.scatter(xs=y_train, zs=x_train['Water Maze CIPL'], ys=x_train['Working Memory CIPL'])
    ax.plot_surface(X=z, Y=yy, Z=xx, color='r',alpha=0.5)
    ax.set_ylabel('Working Memory CIPL')
    ax.set_zlabel('Spatial Memory CIPL')
    ax.set_xlabel('Age (months)')
    props = dict(boxstyle='round', facecolor='g', alpha=0.5)
    ax.text(0.05,0.95,1.0,'age = {0} + {1}(Working) + {2}(Spatial)'.format(str(intercept.round(2)),str(coefficients[0].round(2)),
                                                         str(coefficients[1].round(2))),transform=ax.transAxes,
            fontsize=18,verticalalignment='top',bbox=props,horizontalalignment='left')
    plt.savefig('Results/Regression/OrdinaryLeastSquares/'+title+'.pdf')
    plt.show()



if __name__ == '__main__':
    features = ['Working Memory CIPL','Water Maze CIPL']
    X = data.allData[features]
    y = data.allData['Age']

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    intercept = linreg.intercept_
    coefs = linreg.coef_    # working memory and spatial memory coefficients

    xmax = int(max([X_train['Working Memory CIPL'].max(),X_train['Water Maze CIPL'].max()]))
    xmin = int(min([X_train['Working Memory CIPL'].min(),X_train['Water Maze CIPL'].min()]))
    plot_fit_3D(intercept,coefs,range(xmin-1,xmax+2),X_train,y_train,title='leastSquares')






