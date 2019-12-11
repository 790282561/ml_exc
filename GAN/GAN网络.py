from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

latent_dim = 32
height = 32
width = 32
channels = 3
save_dir = 'pics'

# 构建生成器
generator_input = layers.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyRelu()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')
x = layers.LeakyRelu()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # Conv2DTranspose是Conv2D的反向操作
x = layers.LeakyRelu()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyRelu()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyRelu()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
genenrator = models.Model(generator_input, x)

# 构建判别器
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyRelu()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyRelu()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyRelu()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyRelu()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = models.Model(discriminator_input, x)

discriminator_optimizer = keras.optimizer.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=le - 8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# 组装对抗网络
discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim))
gan_output = discriminator(genenrator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=le - 8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 载入并归一化数据
(train_data, train_labels), (_, _) = mnist.load_data()
train_data = train_data.reshape((-1, 28, 28, 1))
trian_data = train_data.astype('float32') / 255.

# 制造样本标签
valid = np.ones((128, 1))
fake = np.zeros((128, 1))

# 训练生成
for epoch in range(20):
    # 拆分样本
    '''
    从数据集随机挑选128个数据，作为一个批次训练。定义batch_size
    得到的结果是(128, 28, 28, 1)，其实是将样本分解为batch_size
    '''
    idx = np.random.randint(0, train_data.shape[0], 128)
    imgs = train_data[idx]

    # 制造噪音
    '''
    噪音维度(batch_size, 100)，并以此生成图像
    我不理解为什么必须要用正态分布的随机向量生成图像
    '''
    noise = np.random.normal(0, 1, (128, 100))
    gen_img = genenrator.predict(noise)

    # 训练判别器，判别器希望真实图片打上标签1，假图打上0
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_img, fake)

    # 混合真假结果
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 训练生成器
    g_loss = gan.train_on_batch(noise, valid)

    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch, d_loss[0], 100*d_loss[1], g_loss))

    # 每5个epoch保存一个生成图片
    if epoch % 5 == 0:
        gan.save_weights('gan.h5')  # 为什么要保存权重

        img = image.array_to_img(gen_img[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_num' + str(epoch) + '.png'))

        img = image.array_to_img(imgs[0] *255., scale=False)
        img.save(os.path.join(save_dir, 'real_num' + str(epoch) + '.png'))