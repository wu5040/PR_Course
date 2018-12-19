import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def loadDataSet(intX):
    '''
        创建实验样本集,intX--属性值
    '''
    file_data1 = np.loadtxt(
        r'C:\111aaa\PR_Course\expData\genderdata\MALE.txt', delimiter='\t')
    file_data2 = np.loadtxt(
        r'C:\111aaa\PR_Course\expData\genderdata\FEMALE.txt', delimiter='\t')

    groupList1 = []
    classVec1 = []
    for dataItem in file_data1:
        groupList1.append(dataItem[intX])
        classVec1.append(1)

    groupList2 = []
    classVec2 = []
    for dataItem in file_data2:
        groupList2.append(dataItem[intX])
        classVec2.append(0)

    return groupList1, classVec1, groupList2, classVec2


def createFeatureList(dataSet):
    featureSet = set()
    for item in dataSet:
        featureSet.add(item)
    return list(featureSet)


def bagOfFeature2Vec(featureList, inputSet):
    returnVec = [0]*len(featureList)
    for item in inputSet:
        if item in featureList:
            returnVec[featureList.index(item)] += 1
        else:
            print("the feature: %s is not in my FeatureList!" % item)
    return returnVec


def normFun(x, mu, sigma):
    '''
        normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
    '''
    pdf = np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


def GaussianFun(ListX, listClasses):
    mu_sum = 0
    mu_len = 0
    for i in range(len(ListX)):
        mu_sum += ListX[i]
        mu_len += 1

    mu = mu_sum/mu_len
    print("mu=", mu)

    sigma2_sum = 0
    for i in range(len(ListX)):
        sigma2_sum += (ListX[i]-mu)**2

    sigma2 = sigma2_sum/mu_len
    print("sigma2=", sigma2)

    return mu, sigma2


plt.figure(figsize=(8, 5))

x = []
x.append(np.arange(130, 200, 1))
x.append(np.arange(30, 100, 1))
x.append(np.arange(30, 50, 0.1))
pn = []
pn.append([plt.subplot(321), plt.subplot(322)])
pn.append([plt.subplot(323), plt.subplot(324)])
pn.append([plt.subplot(325), plt.subplot(326)])


for i in range(3):
    mListX, mListClasses, fListX, fListClasses = loadDataSet(i)
    mu = [0, 0]
    sigma2 = [0, 0]

    mu[0], sigma2[0] = GaussianFun(mListX, mListClasses)
    mu[1], sigma2[1] = GaussianFun(fListX, fListClasses)

    # 概率密度函数
    y1 = normFun(x[i], mu[0], sqrt(sigma2[0]))
    y2 = normFun(x[i], mu[1], sqrt(sigma2[1]))
    pn[i][0].plot(x[i], y1, color='b', linewidth=2, label="male")
    pn[i][0].plot(x[i], y2, color='r', linewidth=2, label="female")
    pn[i][0].legend()

    # 后验概率函数
    z1 = y1*0.5/(y1*0.5+y2*0.5)
    z2 = y2*0.5/(y1*0.5+y2*0.5)
    pn[i][1].plot(x[i], z1, color='b', linewidth=2, label="male")
    pn[i][1].plot(x[i], z2, color='r', linewidth=2, label="female")
    pn[i][1].legend()

plt.show()
