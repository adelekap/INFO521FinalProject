import matplotlib.pyplot as plt
import seaborn as sns
import data

# scatter plot of water maze data
sns.lmplot('Trial','Water Maze CIPL',data=data.wmazeData,hue='Age')
plt.xlim(0,25)
plt.show()

# water maze and working memory
sns.jointplot('Working Memory CIPL','Water Maze CIPL',data=data.allData)
plt.show()

