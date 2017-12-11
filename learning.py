import numpy as np
import matplotlib.pyplot as plt
import data

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error


def plot_mses(mses,age,alpha):
    plt.plot([1, 2, 3, 4, 5], mses, color='red')
    plt.xlabel('Polynomial Order')
    plt.ylabel('Error')
    plt.savefig('Results/Regression/Learning/{0}degree{1}.pdf'.format(age,alpha))
    plt.show()


def fit_polynomial(data, title, file, alpha=0):
    X_train, X_test, y_train, y_test = train_test_split(data['Trial'], data['Water Maze CIPL'])
    colors = ['teal', 'yellowgreen', 'gold', 'purple', 'pink', 'brown']
    lw = 2
    mses = []
    for degree in [1, 2, 3, 4, 5]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        model.fit(X_train.values.reshape(-1, 1), y_train)
        if degree == 1:
            line = model.predict(np.linspace(0, 30, 100).reshape(-1, 1))
            intercept = model._final_estimator.intercept_
            coef = model._final_estimator.coef_
        y_plot = model.predict(np.linspace(0, 30, 100).reshape(-1, 1))
        mse = mean_squared_error(y_test, model.predict(X_test.values.reshape(-1, 1)))
        mses.append(mse)
        plt.plot(np.linspace(0, 30, 100), y_plot, color=colors[degree - 1], linewidth=lw,
                 label="degree {0}   test error:{1}".format(degree, mse.round(2)))
    ys = data.groupby('Trial').mean()['Water Maze CIPL']
    plt.scatter(range(1, 25), ys, color='black', edgecolors='black', s=30, marker='o',
                label="Average trial performance")
    plt.legend(loc=2, prop={'size': 8})
    plt.title(title)
    plt.xlim(0, 30)
    plt.ylim(0, 60)
    plt.xlabel('Trial')
    plt.ylabel('CIPL')
    plt.savefig('Results/Regression/Learning/' + file)
    plt.show()
    return line,intercept,coef,mses


def fit_line(y,m,o,young,middle,old):
    xs = np.linspace(0,30,100)
    plt.scatter(range(1,25),young.groupby('Trial').mean()['Water Maze CIPL'],color='black', edgecolors='black', s=30, marker='o',
                label="young average trial performance")
    plt.scatter(range(1,25),middle.groupby('Trial').mean()['Water Maze CIPL'],color='gray', edgecolors='black', s=30, marker='o',
                label="middle average trial performance")
    plt.scatter(range(1,25),old.groupby('Trial').mean()['Water Maze CIPL'],color='white', edgecolors='black', s=30, marker='o',
                label="old average trial performance")
    plt.plot(xs,y[0],label='young: y = {0}x+{1}'.format(y[2][1].round(1),y[1].round(1)),color='#440154')
    plt.plot(xs,m[0],label='middle-aged y = {0}x+{1}'.format(m[2][1].round(1),m[1].round(1)),color='#21918c')
    plt.plot(xs,o[0],label='old y = {0}x+{1}'.format(o[2][1].round(1),o[1].round(1)),color='#fde725')
    plt.legend(loc='upper right', prop={'size': 8})
    plt.xlabel('Trial')
    plt.ylabel('CIPL')
    plt.xlim(1, 30)
    plt.ylim(0, 45)
    plt.title('Spatial Learning')
    plt.savefig('Results/Regression/Learning/SpatialLearning.pdf')
    plt.show()

    plot_mses(y[3],'young',alpha)
    plot_mses(m[3],'middle',alpha)
    plot_mses(o[3],'old',alpha)


if __name__ == '__main__':

    young = data.wmazeData[data.wmazeData['Age'] == 6]
    middle = data.wmazeData[data.wmazeData['Age'] == 15]
    old = data.wmazeData[data.wmazeData['Age'] == 23]

    alpha = .02

    fit_line(fit_polynomial(young, 'Young Spatial Learning - alpha = {0}'.format(alpha), '{0}Young.pdf'.format(alpha)),
             fit_polynomial(middle, 'Middle-Aged Spatial Learning - alpha = {0}'.format(alpha), '{0}Middle.pdf'.format(alpha)),
             fit_polynomial(old, 'Old Spatial Learning - alpha = {0}'.format(alpha), '{0}Old.pdf'.format(alpha)),young,middle,old)
