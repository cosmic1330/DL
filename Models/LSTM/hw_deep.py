from numpy import array
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from numpy.random import seed
a = seed(1)


font = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}

plt.rc('font', **font)

n_timestamp = 10
train_days = 1500  # number of days to train from
testing_days = 500  # number of days to be predicted
n_epochs = 25
n_batch_size = 32

#  下載資料
url = "https://raw.githubusercontent.com/danrustia11/WeatherLSTM/master/data/weather_temperature_yilan.csv"
dataset = pd.read_csv(url)
# 2196筆資料
print(len(dataset))


# 資料過濾
# 中值滤波: 能有效抑制噪声，通过把数字图像中一点的值用该点周围的各点值的中位数来代替，让这些值接近，以消除原数据（图像or时序数据）中的噪声。
dataset['Temperature'] = medfilt(dataset['Temperature'], 3)
# 高斯平滑
dataset['Temperature'] = gaussian_filter1d(dataset['Temperature'], 1.2)


# 切分資料
train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)
# iloc[]是用index位置來取我們要的資料(只拿Temperature)
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values

# 設定標準化方法
sc = MinMaxScaler(feature_range=(0, 1)) # 最小最大值標準化

# 是先對partData作fit()的功能，找到該partData的整體統計特性之指標，如平均值、標準差、最大最小值等等(能依據不同目的套用這些指標在不同的轉換(即後面的transform()動作)上，再實行transform(partData)以對partData進行標準化
# 必須先用fit_transform(partData)，之後再transform(restData)
# 執行標準化
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)

# 將資料整理成參考前五天資料，並預測下一天氣溫
def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# 使用模型類別
model_type = 3

if model_type == 1:
    # Single cell LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='relu',
              input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
if model_type == 2:
    # Stacked LSTM （多一層隱藏層）
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
if model_type == 3:
    # Bidirectional LSTM （雙向）
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(50, activation='relu'),
            input_shape=(X_train.shape[1],
                         1)))
    model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size)
loss = history.history['loss']
epochs = range(len(loss))
print(list(history.history))

y_predicted = model.predict(X_test)

y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()


plt.figure(figsize=(8, 7))

plt.subplot(3, 1, 1)
plt.plot(dataset['Temperature'], color='black', linewidth=1, label='True value')
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("All data")


plt.subplot(3, 2, 3)
plt.plot(y_test_descaled, color='black', linewidth=1, label='True value')
plt.plot(y_predicted_descaled, color='red',  linewidth=1, label='Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (n days)")

plt.subplot(3, 2, 4)
plt.plot(y_test_descaled[0:75], color='black', linewidth=1, label='True value')
plt.plot(y_predicted_descaled[0:75], color='red', label='Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (first 75 days)")

plt.subplot(3, 3, 7)
plt.plot(epochs, loss, color='black')
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")

plt.subplot(3, 3, 8)
plt.plot(y_test_descaled-y_predicted_descaled, color='black')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 3, 9)
plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse, 2)))
print("r2=" + str(round(r2, 2)))
