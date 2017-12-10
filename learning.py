import numpy as np
import matplotlib.pyplot as plt
import data

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fit_polynomial(data,title,file,alpha=0):
    X_train, X_test, y_train, y_test = train_test_split(data['Trial'], data['Water Maze CIPL'])
    colors = ['teal', 'yellowgreen', 'gold', 'purple','pink','brown']
    lw = 2

    for degree in [1, 2, 3, 4,5,6]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        model.fit(X_train.values.reshape(-1, 1), y_train)
        y_plot = model.predict(np.linspace(0, 30, 100).reshape(-1, 1))
        mse = mean_squared_error(y_test,model.predict(X_test.values.reshape(-1,1)))
        plt.plot(np.linspace(0, 30, 100), y_plot, color=colors[degree - 1], linewidth=lw,
                 label="degree {0}   test error:{1}".format(degree,mse.round(2)))
    ys = data.groupby('Trial').mean()['Water Maze CIPL']
    plt.scatter(range(1, 25), ys, color='black', edgecolors='black', s=30, marker='o', label="Average trial performance")
    plt.legend(loc = 2,prop={'size': 8})
    plt.title(title)
    plt.xlim(0,30)
    plt.ylim(0,60)
    plt.xlabel('Trial')
    plt.ylabel('CIPL')
    plt.savefig('Results/Regression/Learning/'+file)
    plt.show()

young = data.wmazeData[data.wmazeData['Age'] == 6]
middle = data.wmazeData[data.wmazeData['Age'] == 15]
old = data.wmazeData[data.wmazeData['Age'] == 23]

for alpha in [0,0.01,0.1]:
    fit_polynomial(young,'Young Spatial Learning - alpha = {0}'.format(alpha),'{0}Young.pdf'.format(alpha))
    fit_polynomial(middle,'Middle-Aged Spatial Learning - alpha = {0}'.format(alpha),'{0}Middle.pdf'.format(alpha))
    fit_polynomial(old,'Old Spatial Learning - alpha = {0}'.format(alpha),'{0}Old.pdf'.format(alpha))