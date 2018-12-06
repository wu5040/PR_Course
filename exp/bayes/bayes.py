import numpy as np

def loadDataSet():
    file_data1 = np.loadtxt(
        r'C:\Users\Hell-o\Desktop\模式识别\实验数据\genderdata\boy.txt', delimiter='\t')
    file_data2 = np.loadtxt(
        r'C:\Users\Hell-o\Desktop\模式识别\实验数据\genderdata\girl.txt', delimiter='\t')

    groupList = []
    classVec = []
    for dataItem in file_data1:
        groupList.append(dataItem[0])
        classVec.append(1)
    for dataItem in file_data2:
        groupList.append(dataItem[0])
        classVec.append(0)
    print(groupList, classVec)

    return groupList,classVec


def createFeatureList(dataSet):
    featureSet = set()
    for item in dataSet:
        featureSet.add(item)
    return list(featureSet)

def setOfFeature2Vec(featureList,inputSet):
    returnVec=[0]*len(featureList)
    for item in inputSet:
        if item in featureList:
            returnVec[featureList.index(item)]=1
        else:
            print("the feature: %s is not in my FeatureList!" % item)
    return returnVec

listOPosts,listClasses=loadDataSet()
myFeatureList=createFeatureList(listOPosts)
print(listOPosts)
print(myFeatureList)

Feature2Vec=setOfFeature2Vec(myFeatureList,listOPosts)
print(Feature2Vec)



