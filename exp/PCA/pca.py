import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import decomposition


file_data = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')

# 身高、体重、鞋码
X = file_data[:,:3]
# 性别
y = file_data[:, 3]

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

y_unique=np.unique(y)
colors=['r','b']
for this_y,color in zip(y_unique,colors):
    this_X=X[y==this_y]
    plt.scatter(this_X[:,0],this_X[:,1],c=color)
plt.show()
