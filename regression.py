import matplotlib.pyplot as plt
import seaborn as sns
import data

# scatter plot of water maze data
sns.lmplot('Trial','Water Maze CIPL',data=data.wmazeData,hue='Age',legend=False)
plt.xlim(0,30)
plt.ylim(-3,70)
plt.legend(loc=1)
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

