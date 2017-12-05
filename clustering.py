import data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np

dir = 'Figures/'

def set_colors(data,model,age=True):
    data['Group'] = model.labels_
    column_name = 'Group'
    if age:
        data.loc[data.Group == 2, column_name] = '#00b33c'
        data.loc[data.Group == 0, column_name] = '#4da6ff'
        data.loc[data.Group == 1, column_name] = '#b30000'
    else:
        data.loc[data.Group == 2, column_name] = '#b30000'
        data.loc[data.Group == 0, column_name] = '#4da6ff'
        data.loc[data.Group == 1, column_name] = '#00b33c'
    return data


def plot_cluster_results(data,title,file):
    colors = list(d['Group'])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(ys=data['Age'], zs=data['Water Maze CIPL'], xs=data['Working Memory CIPL'], c=colors)
    ax.set_xlabel('Working Memory CIPL')
    ax.set_zlabel('Water Maze CIPL')
    ax.set_ylabel('Age')
    plt.title(title,fontsize=12)
    plt.tight_layout()
    plt.savefig(dir+file)
    plt.show()


def plot_cluster_diffs_in_age(data,age):
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x=data['Water Maze CIPL'], y=data['Working Memory CIPL'], c=list(data['Group']))
    ax.set_ylabel('Working Memory CIPL')
    ax.set_xlabel('Water Maze CIPL')
    plt.xlim(0,45)
    plt.ylim(0,30)
    plt.title(age+' Cluster',fontsize=12)
    plt.tight_layout()
    plt.savefig(dir+age+'Cluster.pdf')
    plt.show()



if __name__ == '__main__':
    #### PLOT 3D
    #### MODEL INCLUDES AGE
    d = data.allData[['Age','Water Maze CIPL','Working Memory CIPL']]
    m = KMeans(n_clusters=3,random_state=0).fit(data.allData[['Age','Water Maze CIPL','Working Memory CIPL']])
    plot_cluster_results(set_colors(d,m),title='Behavioral Clustering using K-Means (age included)',file='ageIncluded.pdf')

    #### PLOT 3D
    #### MODEL DOES NOT INCLUDE AGE
    d = data.allData[['Age','Water Maze CIPL','Working Memory CIPL']]
    m = KMeans(n_clusters=3,random_state=0).fit(data.allData[['Water Maze CIPL','Working Memory CIPL']])
    plot_cluster_results(set_colors(d,m,age=False),title='Behavioral Clustering using K-Means (age not included)',
                         file='justData.pdf')

    #### SEE DIFFERENCES IN AGE GROUPS
    yng = d[d['Age'] == 6]
    mid = d[d['Age'] == 15]
    old = d[d['Age'] == 23]
    plot_cluster_diffs_in_age(yng,'yng')
    plot_cluster_diffs_in_age(mid,'mid')
    plot_cluster_diffs_in_age(old,'old')

