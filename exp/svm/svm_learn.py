from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

file_data = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')

dataMat=np.delete(file_data.T,[2,3],axis=0).T
label=file_data[:,3]
# print(dataMat)
# print(label)



