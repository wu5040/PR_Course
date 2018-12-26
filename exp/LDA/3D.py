import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_data1 = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\genderdata\MALE.txt', delimiter='\t')
file_data2 = np.loadtxt(
    r'C:\111aaa\PR_Course\expData\genderdata\FEMALE.txt', delimiter='\t')

boyMat = np.mat(file_data1)
girlMat = np.mat(file_data2)

arrMale = np.array(boyMat.T)
arrFemale = np.array(girlMat.T)


mu1 = np.mean(boyMat, axis=0)
mu2 = np.mean(girlMat, axis=0)

print(mu1, mu2)

# 类内散度矩阵Sw
S1 = 0
S2 = 0
for x in boyMat:
    x = np.mat(x)
    S1 += ((x-mu1).T)*(x-mu1)

for x in girlMat:
    x = np.mat(x)
    S2 += ((x-mu2).T)*(x-mu2)

Sw = S1+S2
print(S1)
print(S2)
print(Sw)

omega_x = Sw.I*((mu1-mu2).T)
omega_o = omega_x.T*((mu1+mu2).T)/2
print(omega_x)
print(mu1+mu2)
print(float(omega_o))

x = np.arange(140, 200, 1)
y = np.arange(30, 90, 1)
x, y = np.meshgrid(x, y)

z = (float(omega_o)-float(omega_x[0])*x-float(omega_x[1])*y)/float(omega_x[2])
fig1 = plt.figure()
ax1 = Axes3D(fig1)

ax1.plot_surface(x, y, z,cmap='rainbow')

ax1.scatter(arrMale[0], arrMale[1], arrMale[2], label="Male")
ax1.scatter(arrFemale[0], arrFemale[1], arrFemale[2], label="Female")
plt.legend()
plt.show()
