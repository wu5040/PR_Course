from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np

file_data = np.loadtxt(
    r'/Users/wsg/Documents/PR_Course/expData/studentdataset.csv', delimiter=',')

# 身高、体重、鞋码
dataMat = file_data[:, 0:3]
# 性别
label = file_data[:, 3]

skf = StratifiedKFold(n_splits=10)

K=[1,2,3,4]

KNClassifiers=(KNeighborsClassifier(n_neighbors=kk) for kk in K)
plt.figure()
plts=(plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224))

for KNC,subplt,kk in zip(KNClassifiers,plts,K):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in skf.split(dataMat, label):
        neigh_ = KNC.fit(dataMat[train_index], label[train_index]).predict(
            dataMat[test_index])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(label[test_index], neigh_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        subplt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    subplt. plot([0, 1], [0, 1], linestyle='--', lw=2,
            color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    subplt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    subplt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    subplt.set_xlabel('False Positive Rate')
    subplt.set_ylabel('True Positive Rate')
    subplt.set_title('K={}'.format(kk))
    subplt.legend()

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])

# plt.set_title('Receiver operating characteristic example')


plt.show()
