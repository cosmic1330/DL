import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


data = load_iris(as_frame = True) ## 讀取資料集
inputs = data.data[['sepal length (cm)', 'sepal width (cm)']]
# inputs['target'] = data.target
labels = data.target

print(labels)

fig = plt.figure(figsize=(8, 6))
plt.scatter(inputs['sepal length (cm)'], inputs['sepal width (cm)'], c = labels, cmap = 'Set1')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(inputs, labels, train_size = 0.8, shuffle = True)

svm_list = []

## 因為讀取的資料為 dataframe
svm_clf_poly = SVC(kernel='poly')
svm_clf_poly.fit(x_train.values, y_train.values)
svm_list.append(svm_clf_poly)

svm_clf_rbf = SVC(kernel='rbf')
svm_clf_rbf.fit(x_train.values, y_train.values)
svm_list.append(svm_clf_rbf)

svm_clf_linear = SVC(kernel='linear')
svm_clf_linear.fit(x_train.values, y_train.values)
svm_list.append(svm_clf_linear)

svm_clf_sigmoid = SVC(kernel='sigmoid')
svm_clf_sigmoid.fit(x_train.values, y_train.values)
svm_list.append(svm_clf_sigmoid)

from sklearn.metrics import confusion_matrix
import seaborn as sn

for svm in zip(svm_list):
  svm_pre = svm.predict(x_test.values)
  print('Accuracy: ', accuracy_score(y_test.values, svm_pre))
  res = confusion_matrix(y_test.values, svm_pre)
  fig = plt.figure(figsize=(8, 6))
  sn.heatmap(res, annot=True, cmap="OrRd", fmt='g')

  x_train = np.array(inputs).reshape(-1, 2)

for counter, svm in enumerate(svm_list):
    fig = plt.figure(figsize=(8, 6))
    # prepare for contour，先畫地板
    resolution = 200
    dx = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), resolution) 
    dy = np.linspace(np.min(x_train[:, 1]), np.max(x_train[:, 1]), resolution)
    dx, dy = np.meshgrid(dx, dy)

    # merge dx, dy to test_x，預測地板類別
    test_x = np.c_[dx.flatten(), dy.flatten()]
    z = svm.predict(test_x)
    z = z.reshape(dx.shape)

    # draw，畫有類別的地板
    plt.contourf(dx, dy, z, alpha=0.2)

    # draw all dataset，畫整個資料集
    colors = ['#FF0', '#000', '#00F']
    for i in [0, 1, 2]:
        idx = labels == i
        plt.plot(x_train[idx, 0], x_train[idx, 1], '*', color=colors[i])
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
    plt.show()