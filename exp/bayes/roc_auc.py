from bayesN import LOO
import numpy as np
import matplotlib.pyplot as plt

res, TP, FN, FP, TN = LOO(0, 1)
res1, TP1, FN1, FP1, TN1 = LOO(0, 2)
res2, TP2, FN2, FP2, TN2 = LOO(1, 2)

m1 = TP+FN
m2 = FP+TN

m3 = TP1+FN1
m4 = FP1+TN1

m5 = TP2+FN2
m6 = FP2+TN2

array = np.array(res)
array = array[array[:, 0].argsort()]

array1 = np.array(res1)
array1 = array1[array1[:, 0].argsort()]

array2 = np.array(res2)
array2 = array2[array2[:, 0].argsort()]


x = 0
y = 0
x_list = [0, ]
y_list = [0, ]

for item in reversed(array):
    if item[1] == 1:
        y = y+1/m1
    else:
        x = x+1/m2
    x_list.append(x)
    y_list.append(y)


x = 0
y = 0
x_list1 = [0, ]
y_list1 = [0, ]

for item in reversed(array1):
    if item[1] == 1:
        y = y+1/m3
    else:
        x = x+1/m4
    x_list1.append(x)
    y_list1.append(y)


x = 0
y = 0
x_list2 = [0, ]
y_list2 = [0, ]

for item in reversed(array2):
    if item[1] == 1:
        y = y+1/m5
    else:
        x = x+1/m6
    x_list2.append(x)
    y_list2.append(y)

# 求AUC
sum = 0
for i in range(len(x_list)-1):
    sum = sum+(x_list[i+1]-x_list[i])*(y_list[i]+y_list[i+1])
AUC = 0.5*sum
print("AUC=", AUC)

sum = 0
for i in range(len(x_list1)-1):
    sum = sum+(x_list1[i+1]-x_list1[i])*(y_list1[i]+y_list1[i+1])
AUC1 = 0.5*sum
print("AUC1=", AUC1)

sum = 0
for i in range(len(x_list2)-1):
    sum = sum+(x_list2[i+1]-x_list2[i])*(y_list2[i]+y_list2[i+1])
AUC2 = 0.5*sum
print("AUC2=", AUC2)


# 绘制ROC曲线
plt.figure('ROC')
ax = plt.gca()

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x_list, y_list, color='r', linewidth=2, alpha=1,
        label="H-W (AUC= %0.5f )" % AUC)
# ax.scatter(x_list, y_list, c='b', s=2, alpha=1)
ax.plot(x_list1, y_list1, color='b', linewidth=2, alpha=1,
        label="H-S (AUC= %0.5f )" % AUC1)
# ax.scatter(x_list1, y_list1, c='b', s=2, alpha=1)
ax.plot(x_list2, y_list2, color='y', linewidth=2, alpha=1,
        label="W-S (AUC= %0.5f )" % AUC2)
# ax.scatter(x_list2, y_list2, c='b', s=2, alpha=1)

plt.legend()
plt.show()
