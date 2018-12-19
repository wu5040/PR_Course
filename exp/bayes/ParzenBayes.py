from bayesN import loadDataSet
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


def pnFunc(x, h, Vn, lenData, dataList):
    '''
        Parzen窗法求概率密度函数
    '''
    sum = 0
    for i in range(lenData):
        sum = sum+1/Vn*(
            np.exp(-0.5*((abs(x-dataList[i][0]))**2/h))/sqrt(2*np.pi)
        )
    pdf = sum/lenData
    return pdf

lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
    0, 1)

h = 0.5*sqrt(6)
Vn = h/sqrt(lenData1)

x = np.arange(140, 200, 0.1)
y1 = pnFunc(x, h, Vn, lenData1, groupListMale)
y2 = pnFunc(x, h, Vn, lenData2, groupListFemale)

plt.plot(x, y1, color='b', linewidth=3, label="male")
plt.plot(x, y2, color='r', linewidth=3, label="female")
plt.legend()
plt.show()
