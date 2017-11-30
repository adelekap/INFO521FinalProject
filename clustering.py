import data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.markers as markers
from mpl_toolkits.mplot3d import Axes3D

def plot_cluster_results(Data,model):
    data_labeled = Data
    data_labeled['Group'] = model.labels_

    column_name = 'Group'
    data_labeled.loc[data_labeled.Group == 1, column_name] = 'Low'
    data_labeled.loc[data_labeled.Group == 0, column_name] = 'Middle'
    data_labeled.loc[data_labeled.Group == 2, column_name] = 'High'

    pal = sns.color_palette(['#00b33c', '#4da6ff', '#b30000'])
    sns.lmplot('Trial', 'Water Maze CIPL', data=data_labeled, hue='Group', fit_reg=False, palette=pal)
    #sns.pairplot(Data, hue="Group", palette=pal, diag_kind="kde", size=2.5)
    plt.show()

#### PLOT 2D
#### MODEL JUST DATA NO AGE
plot_cluster_results(data.wmazeData[['Trial','Water Maze CIPL']],
                     KMeans(n_clusters=3, random_state=0).fit(data.wmazeData[['Trial', 'Water Maze CIPL']]))

#### PLOT 3D
#### MODEL INCLUDES AGE
d = data.allData[['Age','Water Maze CIPL','Working Memory CIPL']]
m = KMeans(n_clusters=3,random_state=0).fit(data.allData[['Age','Water Maze CIPL','Working Memory CIPL']])
d['Group'] = m.labels_
column_name = 'Group'
d.loc[d.Group == 2, column_name] = '#b30000'
d.loc[d.Group == 0, column_name] = '#4da6ff'
d.loc[d.Group == 1, column_name] = '#00b33c'

colors=list(d['Group'])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(ys=d['Age'],zs=d['Water Maze CIPL'],xs=d['Working Memory CIPL'],c=colors)
ax.set_xlabel('Working Memory CIPL')
ax.set_zlabel('Water Maze CIPL')
ax.set_ylabel('Age')
plt.savefig('ageIncluded.pdf')
plt.show()

#### PLOT 3D
#### MODEL DOES NOT INCLUDE AGE
d = data.allData[['Age','Water Maze CIPL','Working Memory CIPL']]
m = KMeans(n_clusters=3,random_state=0).fit(data.allData[['Water Maze CIPL','Working Memory CIPL']])
d['Group'] = m.labels_
column_name = 'Group'
d.loc[d.Group == 1, column_name] = '#b30000'
d.loc[d.Group == 0, column_name] = '#4da6ff'
d.loc[d.Group == 2, column_name] = '#00b33c'

colors=list(d['Group'])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(ys=d['Age'],zs=d['Water Maze CIPL'],xs=d['Working Memory CIPL'],c=colors)
ax.set_xlabel('Working Memory CIPL')
ax.set_zlabel('Water Maze CIPL')
ax.set_ylabel('Age')
plt.savefig('justData.pdf')
plt.show()


print('debug')
