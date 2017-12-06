import matplotlib.pyplot as plt
import seaborn as sns
import data
from sklearn import linear_model

sns.pairplot(data.allData,x_vars=['Water Maze CIPL','Working Memory CIPL'],y_vars='Age')
plt.show()

# scatter plot of water maze data
sns.lmplot('Trial','Water Maze CIPL',data=data.wmazeData,hue='Age',legend_out=True)
plt.xlim(0,25)
plt.ylim(-3,70)
plt.title('Water Maze Task Performance by Age',fontsize=12)
plt.tight_layout()
plt.savefig('Figures/waterMazeRegression.pdf')
plt.show()

# water maze and working memory (SHOWS WEAK CORRELATION OF TASK PERFORMANCE)
sns.jointplot('Working Memory CIPL','Water Maze CIPL',data=data.allData,kind='reg')
plt.title('Relationship Between Spatial and Working Memory Tasks',fontsize=12)
plt.tight_layout()
plt.savefig('Figures/taskRegression.pdf')
plt.show()


reg = linear_model.LinearRegression()
x = data.allData['Working Memory CIPL']
y = data.allData['Water Maze CIPL']
reg.fit(data.allData['Working Memory CIPL'],data.allData['Water Maze CIPL'])
print(reg.coef_)


