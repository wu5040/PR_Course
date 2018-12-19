from bayesN import loadDataSet
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pnFunc(x, h, Vn, lenData, dataList):
    '''
        Parzen窗法求概率密度函数
    '''
    sum = 0
    for i in range(lenData):
        sum = sum + \
            (np.exp(-0.5*((abs(x-dataList[i][0]))**2/h))/sqrt(2*np.pi))/Vn
    pdf = sum/lenData
    return pdf


def pnFunc2(X, h, Vn, lenData, dataList):
    dataMat = np.mat(dataList)
    sum = 0
    for i in range(lenData):
        sum += (np.exp((-0.5 *
                        ((X-dataMat[i])*(X-dataMat[i]).T)) / (h*h))/sqrt(2*np.pi))/Vn
    pdf = sum/lenData
    return pdf


lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
    0, 1)

h1 = [0.25,1,5]
# *sqrt(lenData1)

# *sqrt(lenData2)

Vn1 = h1[0]/sqrt(lenData1)
Vn2 = h1[0]/sqrt(lenData2)

Vn3 = h1[1]/sqrt(lenData1)
Vn4 = h1[1]/sqrt(lenData2)

Vn5 = h1[2]/sqrt(lenData1)
Vn6 = h1[2]/sqrt(lenData2)


x = np.arange(140, 200, 1)
y = np.arange(30, 90, 1)
y1 = pnFunc(x, h1[0], Vn1, lenData1, groupListMale)
y2 = pnFunc(x, h1[0], Vn2, lenData2, groupListFemale)

y3 = pnFunc(x, h1[1], Vn3, lenData1, groupListMale)
y4 = pnFunc(x, h1[1], Vn4, lenData2, groupListFemale)

y5 = pnFunc(x, h1[2], Vn5, lenData1, groupListMale)
y6 = pnFunc(x, h1[2], Vn6, lenData2, groupListFemale)


plt.figure(figsize=(10, 3))

plt1=plt.subplot(131)
plt2=plt.subplot(132)
plt3=plt.subplot(133)



plt1.plot(x, y1, color='b', linewidth=3, label="male")
plt1.plot(x, y2, color='r', linewidth=3, label="female")
plt1.legend()

plt2.plot(x, y3, color='b', linewidth=3, label="male")
plt2.plot(x, y4, color='r', linewidth=3, label="female")
plt2.legend()

plt3.plot(x, y5, color='b', linewidth=3, label="male")
plt3.plot(x, y6, color='r', linewidth=3, label="female")
plt3.legend()

plt.show()


# rx, ry = np.meshgrid(x, y)


# matX=[]
# for j in range(len(y)):
#     for i in range(len(x)):
#         sb=np.array([x[i],y[j]])
#         matX.append(sb)

# matX=np.array(matX)
# print(matX)

# ya = pnFunc2(matX, h1, Vn1, lenData1, groupListMale)
# yb = pnFunc2(matX, h2, Vn2, lenData2, groupListFemale)

# fig1 = plt.figure()
# ax1 = Axes3D(fig1)

# ax1.plot_surface(rx, ry, ya, cmap="rainbow")
# ax1.plot_surface(rx, ry, yb, cmap="rainbow")

# ax1.set_zlabel('Probability')  # 坐标轴
# ax1.set_ylabel('weight')
# ax1.set_xlabel('height')




