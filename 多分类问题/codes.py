from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
import numpy as np

# 导入数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

x_train = train_data[1000:]
val_x_train = train_data[:1000]

y_train = train_labels[1000:]
val_y_train = train_labels[:1000]
# 搭建模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# 编译调试
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# 训练模型
# callbacks_list = [
#     callbacks.EarlyStopping(
#         monitor='val_accuracy',
#         patience=1,),
#     callbacks.ModelCheckpoint(
#         filepath='multilabel_model.h5',
#         monitor='val_loss',
#         save_best_only=True,
#     )
# ]
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=512,
    #callbacks=callbacks_list,
    validation_data=(val_x_train, val_y_train))
# 验证测试
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_loss)