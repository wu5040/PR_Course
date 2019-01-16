import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

file_data = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')


# 身高、体重、鞋码
dataX = file_data[:, 0:2]
# 性别
label = file_data[:, 3]

KNClassifier=KNeighborsClassifier(n_neighbors=1)

Store=np.asarray([dataX[0],dataX[1]])
y_Store=np.asarray([label[0],label[1]])

Grabbag=np.delete(dataX,0,axis=0)
Grabbag=np.delete(Grabbag,1,axis=0)
y_Grabbag=np.delete(label,0,axis=0)
y_Grabbag=np.delete(y_Grabbag,1,axis=0)
print(Store)
print(Grabbag)
for x,y in zip(Grabbag,y_Grabbag):
    cla=KNClassifier.fit(Store, y_Store)
    preRes=cla.predict([x])
    if preRes!=y:
        Grabbag=np.delete(Grabbag,)
    print(preRes)