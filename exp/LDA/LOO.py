import sys
sys.path.append(r'C:\111aaa\PR_Course\exp')
from bayes.bayesN import loadDataSet
import numpy as np
import matplotlib.pyplot as plt

def LOO():
    lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
        0, 1)

    errMale = 0
    resArr = []
    mu2 = np.mean(groupListFemale, axis=0)
    mu2 = np.mat(mu2)
    S2 = 0
    for x in groupListFemale:
        x = np.mat(x)
        S2 += ((x-mu2).T)*(x-mu2)
    for i in range(lenData1):
        testX = np.mat(groupListMale[i])
        group = np.delete(groupListMale, i, axis=0)
        mu1 = np.mean(group, axis=0)
        mu1 = np.mat(mu1)
        S1 = 0
        for x in group:
            x = np.mat(x)
            S1 += ((x-mu1).T)*(x-mu1)
        Sw = S1+S2
        omega_x = Sw.I*((mu1-mu2).T)
        omega_o = omega_x.T*((mu1+mu2).T)/2
        gXY = omega_x.T*testX.T-omega_o
        if gXY < 0:  # 预测错误
            errMale += 1
        resArr.append([gXY, 1])

    errFemale = 0
    mu1 = np.mean(groupListMale, axis=0)
    mu1 = np.mat(mu1)
    S1 = 0
    for x in groupListMale:
        x = np.mat(x)
        S1 += ((x-mu1).T)*(x-mu1)
    for i in range(lenData2):
        testX = np.mat(groupListFemale[i])
        group = np.delete(groupListFemale, i, axis=0)
        mu2 = np.mean(group, axis=0)
        mu2 = np.mat(mu2)
        S2 = 0
        for x in group:
            x = np.mat(x)
            S2 += ((x-mu2).T)*(x-mu2)
        Sw = S1+S2
        omega_x = Sw.I*((mu1-mu2).T)
        omega_o = omega_x.T*((mu1+mu2).T)/2
        gXY = omega_x.T*testX.T-omega_o
        if gXY > 0:  # 预测错误
            errFemale += 1
        resArr.append([gXY, 0])
    return resArr,lenData1-errMale, errMale, errFemale, lenData2-errFemale

res, TP, FN, FP, TN = LOO()
print(TP, FN, FP, TN)
m1 = TP+FN
m2 = FP+TN
array = np.array(res)
array = array[array[:, 0].argsort()]

# print((TP+TN)/(TP+FN+FP+TN)*100,'%')

# x = 0
# y = 0
# x_list = [0, ]
# y_list = [0, ]

# for item in reversed(array):
#     if item[1] == 1:
#         y = y+1/m1
#     else:
#         x = x+1/m2
#     x_list.append(x)
#     y_list.append(y)

# # 求AUC
# sum = 0
# for i in range(len(x_list)-1):
#     sum = sum+(x_list[i+1]-x_list[i])*(y_list[i]+y_list[i+1])
# AUC = 0.5*sum
# print("AUC=", AUC)

# # 绘制ROC曲线
# plt.figure('ROC')
# ax = plt.gca()

# ax.set_xlabel('x')
# ax.set_ylabel('y')

# ax.plot(x_list, y_list, linewidth=2, alpha=1,
#         label="AUC= %0.5f" % AUC)

# plt.legend()
# plt.show()