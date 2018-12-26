from bayes.bayesN import trainNB,classifyNB,loadDataSet

import numpy as np
import matplotlib.pyplot as plt

def getBoundary():
    lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale=loadDataSet(0,1)

    matMale=np.mat(groupListMale).T
    matFemale=np.mat(groupListFemale).T

    arrMale=np.array(matMale)
    arrFemale=np.array(matFemale)

    priorP1,priorP2,mu1,sigma1,mu2,sigma2=trainNB(0,1)

    x=np.arange(140,200,0.5)
    y=np.arange(30,90,0.5)

    jumper=[]

    for i in x:
        if classifyNB(i,y[0],priorP1,priorP2,mu1,sigma1,mu2,sigma2)==0:
            for j in y: 
                result=classifyNB(i,j,priorP1,priorP2,mu1,sigma1,mu2,sigma2)
                if result==1:
                    jumper.append(j)
                    break
        else:
            jumper.append(y[0])
            break

    y=np.asarray(jumper)
    print(len(y))
    x=x[0:len(y)]
    return x,y

# plt.figure()
# plt.plot(x, y, color='g', linewidth=2)
# plt.scatter(arrMale[0],arrMale[1],c='b')
# plt.scatter(arrFemale[0],arrFemale[1],c='r')
# plt.xlim(140,200)
# plt.ylim(30,90)
# plt.show()       

