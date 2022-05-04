# 從 HDF5 檔案中載入模型
model = tf.contrib.keras.models.load_model('my_model.h5')

# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 將模型匯出至 JSON（不含參數）
json_string = model.to_json()

# 將模型匯出至 YAML（不含參數）
yaml_string = model.to_yaml()

# 從 JSON 資料重建模型
model = tf.contrib.keras.models.model_from_json(json_string)

# 從 YAML 資料重建模型
model = tf.contrib.keras.models.model_from_yaml(yaml_string)

# 將參數儲存至 HDF5 檔案（不含模型）
model.save_weights('my_model_weights.h5')

# 從 HDF5 檔案載入參數（不含模型）
model.load_weights('my_model_weights.h5')

# 載入參數至不同的模型中使用
model.load_weights('my_model_weights.h5', by_name = True)