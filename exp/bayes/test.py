from bayesN import LOO
import numpy as np
import matplotlib.pyplot as plt

res,TP,FN,FP,TN=LOO()

m1=TP+FN
m2=FP+TN

array=np.array(res)
array=array[array[:,0].argsort()]


x=0
y=0
x_list=[0,]
y_list=[0,]

for item in reversed(array):
    if item[1]==1:
        y=y+1/m1
    else:
        x=x+1/m2
    x_list.append(x)
    y_list.append(y)

plt.figure('ROC')
ax = plt.gca()

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.5)
ax.scatter(x_list, y_list, c='b', s=2, alpha=1)
plt.show()

# def y_auc(m,x_list,y_list):
#     y_auc=0
#     for i in range(m):
#         y_auc=y_auc+(x_list[i+1]-x_list[i])*(y_list[i]+y_list[i+1])
#     return 0.5*y_auc
