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
# plt.xlabel('sepal length (cm)')｀｀
# plt.ylabel('sepal width (cm)')
# plt.show()

# 3維資料顯示
import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
fig.show()

# 訓練模型
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, train_size = 0.8, shuffle = True)
print(len(x_train)) #120筆
print(len(x_test)) #30筆
print(np.ravel(y_train))

knn_clf = KNeighborsClassifier(n_neighbors = 5)
knn_clf.fit(x_train, np.ravel(y_train)) # np.ravel 是為了將 y_train 

# 預測
knn_pre = knn_clf.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, knn_pre))

# 混淆矩陣
from sklearn.metrics import confusion_matrix
import seaborn as sn
res = confusion_matrix(y_test, knn_pre)
fig = plt.figure(figsize=(8, 6))
sn.heatmap(res, annot=True, cmap="OrRd", fmt='g')
plt.show()

# cross validation
data = load_iris()

inputs = data.data
labels = data.target

n_split = 5
acc_list = []
counter = 1

for train_index, test_index in KFold(n_split).split(inputs): # 每一次的 for，Kold 會幫你切好該次用於訓練和驗證的 Fold
    # 準備此次的資料
    x_train, x_test = inputs[train_index], inputs[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 使用此次訓練資料訓練
    knn_clf = KNeighborsClassifier(n_neighbors = 5)
    knn_clf.fit(x_train, y_train)

    # 使用此次驗證資料驗證
    knn_pre = knn_clf.predict(x_test)
    print('accuracy on {} fold: '.format(counter), accuracy_score(y_test, knn_pre))
    counter = counter + 1
    acc_list.append(accuracy_score(y_test, knn_pre))

    # draw the point that you use to train in this fold
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = 'Set1')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

print('avg acc: ', sum(acc_list)/len(acc_list))