import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 数据加载及预处理
(x_data, x_labels), (y_data, y_labels) = mnist.load_data()
x_data = x_data.astype('float32') / 255 - 0.5  # 对应于tanh的[-1, 1]范围
x_data = x_data.reshape((-1, 784))

y_data = y_data.astype('float32') / 255 - 0.5
y_data = y_data.reshape((-1, 784))

# 构建及编译网络
input_image = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_image)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(2)(encoded)

decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(784, activation='tanh')(decoded)

autoencoder = Model(input_image, decoded_output)
encoder = Model(input_image, encoded_output)
autoencoder.compile(optimizer='adam', loss='mse')

# 训练网络
autoencoder.fit(x_data, x_data, epochs=10, batch_size=512, shuffle=True)

# 预测结果
prediction = autoencoder.predict(y_data)
# 展示压缩结果
x = prediction[:, 0]
y = prediction[:, 1]
plt.scatter(x, y, s=3, c=y_labels)
plt.show()
plt.savefig('自编码压缩为2的坐标点.jpg', dpi=300)