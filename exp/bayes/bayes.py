import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

#创建实验样本集,intX--属性值
def loadDataSet(intX):
    file_data1 = np.loadtxt(
        r'/home/wsg/111aaa/PR_Course/expData/genderdata/boy.txt', delimiter='\t')
    file_data2 = np.loadtxt(
        r'/home/wsg/111aaa/PR_Course/expData/genderdata/girl.txt', delimiter='\t')

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

    return groupList1,classVec1,groupList2,classVec2

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

#normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normFun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def GaussianFun(ListX,listClasses):
    mu_sum=0
    mu_len=0
    for i in range(len(ListX)):
            mu_sum+=ListX[i]
            mu_len+=1

    mu=mu_sum/mu_len
    print("mu=",mu)

    sigma2_sum=0
    for i in range(len(ListX)):
            sigma2_sum+=(ListX[i]-mu)**2

    sigma2=sigma2_sum/mu_len
    print("sigma2=",sigma2)

    return mu,sigma2

def funImg(pn,x,mu,sigma2):

    y1 = normFun(x, mu[0], sqrt(sigma2[0]))
    y2 = normFun(x, mu[1], sqrt(sigma2[1]))


    # 参数,颜色，线宽
    pn.plot(x,y1, color='b',linewidth = 3,label="male")
    pn.plot(x,y2, color='r',linewidth = 3,label="female")

    pn.legend()
    # pn.title('GaussianFun')
    pn.set_xlabel('Feature')
    pn.set_ylabel('Probability')

plt.figure(figsize=(10,10))

x=[]
x.append(np.arange(130,200,1))
x.append(np.arange(30,100,1))
x.append(np.arange(30,50,0.1))
pn=[]
pn.append(plt.subplot(311))
pn.append(plt.subplot(312))
pn.append(plt.subplot(313))


for i in range(3):
    mListX,mListClasses,fListX,fListClasses=loadDataSet(i)
    mu=[0,0]
    sigma2=[0,0]

    mu[0],sigma2[0]=GaussianFun(mListX,mListClasses)
    mu[1],sigma2[1]=GaussianFun(fListX,fListClasses)

    funImg(pn[i],x[i],mu,sigma2)

plt.show()
