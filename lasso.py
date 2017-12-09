import data
from sklearn import linear_model
from sklearn.model_selection import train_test_split,cross_val_score
from LeastSquares import plot_fit_3D
import numpy as np



X=data.allData[['Water Maze CIPL','Working Memory CIPL']]
y = data.allData['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
xmax = int(max([X_train['Working Memory CIPL'].max(), X_train['Water Maze CIPL'].max()]))
xmin = int(min([X_train['Working Memory CIPL'].min(), X_train['Water Maze CIPL'].min()]))

reg = linear_model.Lasso(alpha=1.0)
reg.fit(X_train,y_train)
intercept = reg.intercept_
coefficients = reg.coef_
cv = np.mean(cross_val_score(reg,X_test,y_test,cv=10))
plot_fit_3D(intercept,coefficients,range(xmin-1,xmax+2),X_train,y_train,'Lasso Regression1.0',
            cv,'Results/Regression/Lasso/')

