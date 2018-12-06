import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#创建实验样本集
def loadDataSet():
    file_data1 = np.loadtxt(
        r'/home/wsg/111aaa/PR_Course/expData/genderdata/boy.txt', delimiter='\t')
    file_data2 = np.loadtxt(
        r'/home/wsg/111aaa/PR_Course/expData/genderdata/girl.txt', delimiter='\t')

    groupList = []
    classVec = []
    for dataItem in file_data1:
        groupList.append(dataItem[0])
        classVec.append(1)
    for dataItem in file_data2:
        groupList.append(dataItem[0])
        classVec.append(0)

    return groupList,classVec


def createFeatureList(dataSet):
    featureSet = set()
    for item in dataSet:
        featureSet.add(item)
    return list(featureSet)

def bagOfFeature2Vec(featureList,inputSet):
    returnVec=[0]*len(featureList)
    for item in inputSet:
        if item in featureList:
            returnVec[featureList.index(item)]+=1
        else:
            print("the feature: %s is not in my FeatureList!" % item)
    return returnVec



listOPosts,listClasses=loadDataSet()

print(listOPosts,listClasses)

u_sum=0
u_len=0
for i in range(len(listOPosts)):
    if(listClasses[i]==1):
        u_sum+=listOPosts[i]
        u_len+=1

u=u_sum/u_len
print(u)

o2_sum=0
for i in range(len(listOPosts)):
    if(listClasses[i]==1):
        o2_sum+=(listOPosts[i]-u)**2

o2=o2_sum/u_len
print(o2)

#normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

# x的范围为60-150，以1为单位,需x根据范围调试
x = np.arange(140, 200, 1)

# x数对应的概率密度
y = normfun(x, u, sqrt(o2))

# 参数,颜色，线宽
plt.plot(x,y, color='g',linewidth = 3)

plt.title('IQ distribution')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.show()










