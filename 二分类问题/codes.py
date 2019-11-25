import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# 数据准备
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# train_data = pad_sequences(train_data, maxlen=100, padding='post')
# train_data = to_categorical(train_data)
# test_data = pad_sequences(test_data, maxlen=100, padding='post')
# test_data = to_categorical(test_data)

def vectorize_sequences(seqeunces, dimension=10000):
    results = np.zeros((len(seqeunces), dimension))
    for i, sequence in enumerate(seqeunces):
        results[i, sequence] = 1.
    return results


train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)

train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 模型搭建
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 模型编译
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_acc',
        patience=1,
    ),
    callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

# 模型训练
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    # callbacks=callbacks_list,
    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_acc)