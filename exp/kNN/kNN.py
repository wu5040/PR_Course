from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

file_data = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')

#身高、体重、鞋码
dataMat=file_data[:,0:3]


#性别标签
label=file_data[:,3]



neigh=KNeighborsClassifier(n_neighbors=3)

neigh.fit(dataMat[0:700],label[0:700])

print(label[800:900])
print(neigh.predict(dataMat[800:900]))