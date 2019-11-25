from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

nb_class = 10
nb_epochs = 4
batchsize = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape((-1, 28, 28, 1))  # -1表示数量未知，电脑自行处理
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, nb_class)
y_test = to_categorical(y_test, nb_class)

# 建立模型
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(nb_class, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, epochs=nb_epochs, batch_size=batchsize)
evaluation = model.evaluate(y_train, y_test)
print(evaluation)
