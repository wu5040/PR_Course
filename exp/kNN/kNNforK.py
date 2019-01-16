from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_validate
import matplotlib.pyplot as plt
import numpy as np


file_data = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')

# 身高、体重、鞋码
dataMat = file_data[:, 0:3]
# 性别
label = file_data[:, 3]

skf = StratifiedKFold(n_splits=10)

K = np.arange(1, 101, 1)
KNClassifiers = (KNeighborsClassifier(n_neighbors=kk) for kk in K)
accuracys = []

for KNC, kk in zip(KNClassifiers, K):
    i = 0
    accuracy = 0
    for train_index, test_index in skf.split(dataMat, label):
        neigh_ = KNC.fit(dataMat[train_index], label[train_index]).predict(
            dataMat[test_index])
        err = 0
        for a, b in zip(label[test_index], neigh_):
            if a != b:
                err += 1
        accuracy += 1-err/len(label[test_index])
        i += 1

    mean_accu = accuracy/10
    accuracys.append(mean_accu)

plt.plot(K, accuracys)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()
