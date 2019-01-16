from KL import KL
import numpy as np
from sklearn.datasets import load_breast_cancer,load_digits
import matplotlib.pyplot as plt

# dataSet=load_breast_cancer()
dataSet=load_digits()

kl=KL(dataSet.data,dataSet.target,1)
lambdaArr=kl.getLambdaArr()

print(lambdaArr)

lambdaSum=sum(lambdaArr)

print(lambdaSum)

Loss=[]
for i in range(1,len(lambdaArr)):
    ssum=sum(lambdaArr[i:])
    Loss.append(ssum)

x=np.arange(1,len(lambdaArr),1)
plt.plot(x,Loss,'^',linestyle='-',markerfacecolor='r')
plt.show()

