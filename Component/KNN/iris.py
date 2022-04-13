import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

data = load_iris(as_frame = True) ## 讀取資料集
inputs = data.data
labels = data.target
# print(inputs.head())  #  映出前四筆
# print(inputs.columns) # 所有input欄位名稱

## 以前兩個維度繪製資料點
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(inputs['sepal length (cm)'], inputs['petal width (cm)'], c = labels, cmap = 'Set1')
# plt.xlabel('sepal length (cm)')
# plt.ylabel('sepal width (cm)')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(inputs, labels, train_size = 0.8, shuffle = True)
print(len(x_train)) #120筆
print(len(x_test)) #30筆
print(np.ravel(y_train))

knn_clf = KNeighborsClassifier(n_neighbors = 5)
knn_clf.fit(x_train, np.ravel(y_train)) # np.ravel 是為了將 y_train 



knn_pre = knn_clf.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, knn_pre))


from sklearn.metrics import confusion_matrix
import seaborn as sn
res = confusion_matrix(y_test, knn_pre)
fig = plt.figure(figsize=(8, 6))
sn.heatmap(res, annot=True, cmap="OrRd", fmt='g')
plt.show()